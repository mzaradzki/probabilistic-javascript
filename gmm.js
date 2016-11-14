


var GaussianMixtureModel = function(nbstates, obsdim) {
	var g = [];
  var pi = [];
  for (var i=0; i<nbstates; i++) {
    var covars = numeric.identity(obsdim);
    var means = numeric.add(numeric.mul(covars[0], 0), 3+(i-0.5*nbstates)/nbstates); // to have different vectors
    gaussian = new GaussianLaw(means, covars); // FILL IN MORE ARGUMENTS
    g.push(gaussian);
    pi.push(1/nbstates);
  };
  this.observationProbabilityCPDs = g;
  this.stateDistribution = pi;

  this.numberOfStates = nbstates;
  this.dimensionOfObservations = obsdim;
};

GaussianMixtureModel.prototype.numberOfStates = 0;
GaussianMixtureModel.prototype.dimensionOfObservations = 0;


GaussianMixtureModel.prototype.simulateStates = function(pathLength, withObservations) {

  var states = [];
  var observations = [];

  for (var t=0; t<pathLength; t++) {
    states.push(-1); // dummy assignment to adjust vector length
    var nextprobas = this.stateDistribution;
    var r = Math.random();
    var cumprob = 0;
    for (var s=0; s<this.numberOfStates; s++) {
      cumprob+= nextprobas[s];
      if ((states[t]<0) && (r<cumprob))
      {
        states[t] = s;
      }
    }
    if ((states[t]<0) || (states[t]>=this.numberOfStates))
    {
      throw new Error('states was not assigned properly');
    }
    if (withObservations)
    {
      observations.push(this.observationProbabilityCPDs[states[t]].simulate());
    }
  }
  var dico = {states: states};
  if (withObservations)
  {
    dico.observations = observations;
  }
  return dico
};


GaussianMixtureModel.prototype._softmax = function(observation) {
	var nbstates = this.numberOfStates;
	var probs = [];
	for (var st=0; st<nbstates; st++) {
		probs.push(this.observationProbabilityCPDs[st].pdf(observation));
	}
	return numeric.div(probs, numeric.sum(probs));
}


GaussianMixtureModel.prototype.fitObservations = function(observations, maxIters) {
  var iter, t, j, k;
  var probs;
  var nbstates = this.numberOfStates;
  var obsdim = this.dimensionOfObservations;
  // EM loop
  for (iter=0; iter<maxIters; iter++) {
		var probabilities = [];
		var newStateDistribution = numeric.mul(this.stateDistribution, 0); // null vector of correct dimension
		// so called Expectation step
		for (t=0; t<observations.length; t++) {
			probs = this._softmax(observations[t]);
			probabilities.push(probs);
			newStateDistribution = numeric.add(newStateDistribution, probs);
		};
		// update
		this.stateDistribution = numeric.div(newStateDistribution, numeric.sum(newStateDistribution));
		//
		for (var st=0; st<nbstates; st++) {
			var denom = 0;
			// maximization step : wrt to means
			var newMeans = numeric.mul(this.observationProbabilityCPDs[st].means, 0); // null vector of correct dimension
			for (t=0; t<observations.length; t++) {
				probs = probabilities[t];
				newMeans = numeric.add(newMeans, numeric.mul(probs[st], observations[t]));
				denom = denom + probs[st];
			};
			newMeans = numeric.div(newMeans, denom);
			// maximization step : wrt to covariances
			var newVars = numeric.mul(this.observationProbabilityCPDs[st].covariances, 0); // null matrix of correct dimension
			for (t=0; t<observations.length; t++) {
				probs = probabilities[t];
				for (j=0; j<obsdim; j++) {
					for (k=0; k<obsdim; k++) {
						newVars[j][k] = newVars[j][k] + probs[st] * (observations[t][j]-newMeans[j]) * (observations[t][k]-newMeans[k]);
					}; // k
				}; // j
			}; // i
			newVars = numeric.div(newVars, denom);
			// update
			this.observationProbabilityCPDs[st] = new GaussianLaw(newMeans, newVars);
		}
	}
};


GibbsGMM = function(nbstates, mcmcsteps, observations) {
  // see T.Hastie, R.Tibshirani, J.Friedman : The Elements of Statistical Learning
  // sample the means only at present
  // TO DO : extend to whole simulation
  var obsdim = observations[0].length;
  var gmm0 = new GaussianMixtureModel(nbstates, obsdim);
  gmm0.fitObservations(observations);
  for (var mcstep=0; mcstep<mcmcsteps; mcstep++) {
    console.log('new MCMC step')
    var denoms = numeric.mul(gmm0.stateDistribution, 0); // null vector of correct dimension
    var guesses = [];
    for (var s=0; s<nbstates; s++) {
      guesses.push( numeric.mul(gmm0.stateDistribution, 0) ); // null vector of correct dimension
    }
    for (var t=0; t<observations.length; t++) {
      var probas = gmm0._softmax(observations[t]);
      var activestates = []; // TO DO : random binary vector to simulate
      for (var s=0; s<nbstates; s++) {
        guesses = numeric.add(guesses[s], numeric.mul(observations[i], activestates[s]));
      }
      denoms = numeric.add(denoms, activestates);
    }
    for (var s=0; s<nbstates; s++) {
      guesses = numeric.div(guesses[s], denoms[s]);
    }
    for (var s=0; s<nbstates; s++) {
      var covariances = gmm0.observationProbabilityCPDs[s].covariances
      var posterior = new GaussianLaw(guesses[s], covariances);
      var sampledmeans = posterior.simulate();
      gmm0.observationProbabilityCPDs[s] = new GaussianLaw(sampledmeans, covariances);
      console.log(sampledmeans);
    }
  }
};
