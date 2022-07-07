# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# main target to create the html docs. To create PDF or Latex docs
# copy this target and replace `html` with `pdf` / `latex` where neccesary.
# html: source/notebooks/AdvancedParametrisation
# 	sphinx-build -M html source build

# source/notebooks/1_advancedParametrisation.rst: ../examples/advanced_parametrisation.ipynb
# 	./notebook_converter.sh ../examples/advanced_parametrisation.ipynb
# 	./link_converter.sh source/notebooks/advanced_parametrisation.rst AdvancedParametrisation

# NDK_PATH := $ ln -s /openqaoa/examples ./notebooks

# html: source/notebooks/1_workflows_example.rst
# 		# source/notebooks/1_workflows_example.rst
# 		# source/notebooks/2_simulators_comparison.rst
# 		# source/notebooks/3_qaoa_on_qpus.rst
# 		# source/notebooks/4_qaoa_variational_parameters.rst
# 		# source/notebooks/5_advanced_parameterization.rst
# 		# source/notebooks/6_fast_qaoa_simulator.rst
# 		# source/notebooks/7_cost_landscapes_w_manual_mode.rst
# 		# source/notebooks/8_results_example.rst
# 		# source/notebooks/9_RQAOA_example.rst
# 	sphinx-build -M html source build


# source/notebooks/1_workflows_example.rst:
# 	./notebook_converter.sh

clean:
	rm -r source/notebooks/
	sphinx-build -M clean source build

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
