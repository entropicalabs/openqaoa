pipeline {
  agent {
    node {
      label 'SYS-JENKINS-NODE-1'
    }

  }
  stages {
    stage('error') {
      agent any
      steps {
        git(url: 'https://github.com/entropicalabs/qaoa-qubit-routing.git', branch: 'main')
      }
    }

  }
}