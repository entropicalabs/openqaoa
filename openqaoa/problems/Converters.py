#   Copyright 2022 Entropica Labs
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
from collections import defaultdict
from ..problems.problem import QUBO

class FromDocplex2IsingModel:
    
    def __init__(self, model):
        
        
        """
        Creates an instance to translate Docplex models to its Ising Model representation

        Parameters
        ----------
        model : Docplex model
            It is an object that has the mathematical expressions (Cost function,
            Inequality and Equality constraints) of an optimization problem in
            binary representation.

        """
        # assign the docplex Model
        self.model = model.copy()
        
        # save the index in a dict
        self.idx_terms = {}
        for x in self.model.iter_variables():
            self.idx_terms[x] = x.index

    def linear_expr(self, expr):
        """
        Adds linear expresions to the objective function stored in self.qubo_dict.

        Parameters
        ----------
        expr : model.objective_expr
            A docplex model attribute.

        """
        for x, weight in expr.get_linear_part().iter_terms():
            self.qubo_dict[(self.idx_terms[x], )] += weight
                

    def quadratic_expr(self, expr):
        """
        Adds quadratic terms to the objective function stored in self.qubo_dict.

        Parameters
        ----------
        expr : model.objective_expr
            A docplex model attribute.
        """

        #save the terms and coeffs of the quadratic part, 
        #considering two index i and j 
        for x, y, weight in expr.iter_quad_triplets():
            i = self.idx_terms[x]
            j = self.idx_terms[y]
            if i == j: # if this is the term is  Z1**2 for example
                self.qubo_dict[(i, )] += weight
            else:
                self.qubo_dict[(i, j)] += weight

    def Equality2Penalty(self, expression, multiplier:float):
        """
        Add equality constraints to the cost function using the penality representation.
        The constraints should be linear.
        
        Parameters
        ----------
        expression : docplex.mp.linear.LinearExpr
            docplex equality constraint "expression == 0".
        multiplier : float
            Lagrange multiplier of the penalty

        Returns
        -------
        penalty : docplex.mp.quad.QuadExpr
            Penalty that will be added to the cost function.

        """
        penalty = multiplier * (expression) ** 2
        return penalty
    
    def bounds(self, linear_expression):
        """
        Generates the limits of a linear term

        Parameters
        ----------
        linear_exp : generator
            Iterable term with the quadratic terms of the cost function.

        Returns
        -------
        l_bound : float
            Lower bound of the quadratic term.
        u_bound : float
            Upper bound of the quadratic term
        """
        
        l_bound = u_bound = linear_expression.constant # Lower and upper bound of the contraint
        for term, coeff in linear_expression.iter_terms():
            l_bound += min(0, coeff)
            u_bound += max(0, coeff)
            
        return l_bound, u_bound
    
    def quadratic_bounds(self, iter_exp):
        """
        Generates the limits of the quadratic terms

        Parameters
        ----------
        iter_exp : generator
            Iterable term with the quadratic terms of the cost function.

        Returns
        -------
        l_bound : float
            Lower bound of the quadratic term.
        u_bound : float
            Upper bound of the quadratic term

        """
        l_bound = 0
        u_bound = 0
        for z1, z2, coeff in iter_exp:
            l_bound += min(0, coeff)
            u_bound += max(0, coeff)
        return l_bound, u_bound
        
    def Inequaility2Equality(self, constraint):
        """
        Transform inequality contraints into equality constriants using 
        slack variables.

        Parameters
        ----------
        constraint : docplex.mp.constr.LinearConstraint
            DESCRIPTION.

        Returns
        -------
        new_exp : docplex.mp.linear.LinearExpr
            The equality constraint representation of the inequality constraint.

        """
        
        if constraint.sense_string == "LE":
            new_exp = constraint.get_right_expr() + -1 * constraint.get_left_expr()
        elif constraint.sense_string == "GE":
            new_exp = constraint.get_left_expr() + -1 * constraint.get_right_expr()
        else:
            print("Not recognized constraint.")
            
        lower_bound , upper_bound = self.bounds(new_exp)
        slack_lim = upper_bound # Slack var limit

        if slack_lim > 0:
            sign = - 1
        else:
            sign = 1

        n_slack = int(np.ceil(np.log2(abs(slack_lim + 1))))
        if n_slack > 0:
            slack_vars = self.model.binary_var_list(n_slack, name=f"slack_{constraint.name}")
            for x in slack_vars:
                self.idx_terms[x] = x.index
            
            for nn, var in enumerate(slack_vars[:-1]):
                new_exp += sign * (2 ** nn) * var
    
            new_exp += sign * (slack_lim - 2 ** (n_slack - 1) + 1) * slack_vars[-1] # restrict the last term to fit the upper bound

        return new_exp

    def multipliers_generators(self):
        """
        Penality term size adapter, this is the Lagrange multiplier of the cost
        function penalties for every constraint if the multiplier is not indicated
        by the user. 

        Returns
        -------
        float
            the multiplier resized by the cost function limits.

        """
        cost_func = self.model.objective_expr
        l_bound_linear, u_bound_linear = self.bounds(cost_func.get_linear_part())
        l_bound_quad, u_bound_quad = self.quadratic_bounds(cost_func.iter_quad_triplets())
        return 1.0 + (u_bound_linear - l_bound_linear) + (u_bound_quad - l_bound_quad)

    def linear_constraints(self, multipliers=None) -> None:
        """
        Adds the constraints of the problem to the objective function. 

        Parameters
        ----------
        multiplier : List
            For each constraint a multiplier, if None it automatically is selected.

        Returns
        -------
        None.

        """

        constraints_list = list(self.model.iter_linear_constraints())
        n_constraints = len(constraints_list)

        if multipliers == None:
            multipliers = n_constraints * [self.multipliers_generators()]

        elif type(multipliers) in [int, float]:
            multipliers = n_constraints * [multipliers]
        
        elif type(multipliers) != list:
            print(f"{type(multipliers)} is not a accepted format")
            
        for cn, constraint in enumerate(constraints_list):
            if constraint.sense_string  == "EQ": #Equality constraint added as a penalty.
                left_exp = constraint.get_left_expr()
                right_exp = constraint.get_right_expr()
                expression = left_exp + -1 * right_exp
                penalty = self.Equality2Penalty(expression, multipliers[cn])
            elif constraint.sense_string in ["LE", "GE"]: # Inequality constraint added as a penalty with additional slack variables.
                constraint.name = f"C{cn}"
                ineq2eq = self.Inequaility2Equality(constraint)
                penalty = self.Equality2Penalty(ineq2eq, multipliers[cn])
            else:
                print("Not an accepted constraint!")
            
            self.linear_expr(penalty)
            self.quadratic_expr(penalty)
            self.constant += penalty.constant


    def qubo_to_ising(self, n_variables, qubo_terms, qubo_weights):
        """
        Converts the terms and weights in QUBO representation ([0,1])
        to the Ising representation ([-1, 1]). 

        Parameters
        ----------
        n_variables : int
            number of variables.
        qubo_terms : List
            List of QUBO variables.
        qubo_weights : List
            coefficients of the variables
            

        Returns
        -------
        Ising Model stored on QUBO class

        """
        ising_terms, ising_weights = [], []
        linear_terms = np.zeros(n_variables)
        
        constant_term = 0
        # Process the given terms and weights
        for weight, term in zip(qubo_weights, qubo_terms):

            if len(term) == 2:
                u, v = term

                if u != v:
                    ising_terms.append([u, v])
                    ising_weights.append(weight / 4)
                else:
                    constant_term += weight / 4
                
                linear_terms[term[0]] -= weight / 4
                linear_terms[term[1]] -= weight / 4
                constant_term += weight / 4
            elif len(term) == 1:
                linear_terms[term[0]] -= weight / 2
                constant_term += weight / 2
            elif len(term) == 0:
                constant_term += weight
            else:
                print(f"Term {term} is not recognized!")
        
        for variable, linear_term in enumerate(linear_terms):
            ising_terms.append([variable])
            ising_weights.append(linear_term)
        
        ising_terms.append([])
        ising_weights.append(constant_term)

        return QUBO(n_variables, ising_terms, ising_weights)

    def get_ising_model(self, multipliers: float = None):
        """
        Creates a Ising Model form a Docplex model

        Parameters
        ----------
        model : Docplex model
            It is an object that has the mathematical expressions (Cost function,
            Inequality and Equality constraints) of an optimization problem.

        Returns
        -------
        The Ising encoding of this problem.

        """
        # save a dictionary with the qubo information
        self.qubo_dict = defaultdict(float)
        
        # objective sense
        if self.model.objective_sense.is_minimize():
            #valid the expression is an equation
            self.objective_expr = self.model.objective_expr
        else:
            #valid the expression is an equation
            self.objective_expr = -1 * self.model.objective_expr
        
        
        # Obtain the constant from the model
        self.constant = self.objective_expr.constant
        
        # Save the terms and coeffs form the linear part
        self.linear_expr(self.objective_expr)
            
        # Save the terms and coeffs form the quadratic part
        self.quadratic_expr(self.objective_expr)
        
        # Add the linear constraints into the qubo
        self.linear_constraints(multipliers=multipliers)

        terms = list(self.qubo_dict.keys()) + [[]] # The right term is for adding the constant part of the QUBO 
        
        weights = list(self.qubo_dict.values()) + [self.constant]
        
        n_variables = self.model.number_of_variables
            
        #convert the docplex terms in a Hamiltonian
        return  self.qubo_to_ising(n_variables, terms, weights)
