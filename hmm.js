
// INFORMATION
// modified from this source code : https://github.com/timdream/hmm
// - fixed some indices
// - extended to cover multi-variate gaussian observations

// TO DO
// - add method to generate smart starting point for Baum-Welch based on observations vector
// - add method to randomize starting point for Baum-Welch
// - check implementation of Viterbi algorithm
// - test and debug method GaussianLaw.simulate and HiddenMarkovModel.simulateStates

// TEST CODE
// var hmm = new HiddenMarkovModel(2,2)
// hmm.fitObservations([[1.1,2.1],[1,2.3],[1.5,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[0.3,-1],[0,-1.3],[0.4,-1.1],[0.4,-1.8],[0,-1.4],[0.5,-1.3],[0,-1],[0,-1],[0.2,-1.5],[0.8,-1.5],[0,-1],[0,-1]], 10)

var HiddenMarkovModel = function(nbstates, obsdim) {
  //var a = this.stateTransitionMatrix =
  //  new HMMMatrices.HMMStateTransitionMatrix(aDef);
  var transmat = numeric.add(numeric.identity(nbstates), 0.1);
  this.stateTransitionMatrix = numeric.div(transmat, numeric.sum(transmat[0]));
  /*var b = this.observationProbabilityMatrix =
    new HMMMatrices.HMMObservationProbabilityMatrix(bDef);*/
  var g = [];
  var pi = [];
  for (var i=0; i<nbstates; i++) {
    var covars = numeric.identity(obsdim);
    var means = numeric.add(numeric.mul(covars[0], 0), 3+(i-0.5*nbstates)/nbstates); // to have different vectors
    var gaussian = new GaussianLaw(means, covars); // FILL IN MORE ARGUMENTS
    g.push(gaussian);
    pi.push(1/nbstates);
  };
  this.observationProbabilityCPDs = g;
  this.initialStateDistributionMatrix = [pi]; // WARNING : nest in matrix for legacy reasons

  /*if (a.numberOfStates !== b.numberOfStates ||
      a.numberOfStates !== pi.numberOfStates) {
    throw new Error('HiddenMarkovModel: number of states mismatch.');
  }*/

  this.numberOfStates = nbstates;
  //this.numberOfObservationSymbols = b.numberOfObservationSymbols;
  this.dimensionOfObservations = obsdim;
};

HiddenMarkovModel.prototype.numberOfStates = 0;
//HiddenMarkovModel.prototype.numberOfObservationSymbols = 0;
HiddenMarkovModel.prototype.dimensionOfObservations = 0;

/*
// This is (7) but calculated with log() to prevent underflow.
HiddenMarkovModel.prototype.getProbabilityOfStateSequenceForObservations =
function(x, o) {
  if (x.length !== o.length) {
    throw new Error('HiddenMarkovModel: length mismatch.');
  }

  var a = this.stateTransitionMatrix;
  var b = this.observationProbabilityMatrix;
  var pi = this.initialStateDistributionMatrix;

  var p = Math.log(pi[0][x[0]] * b[x[0]][o[0]]);

  for (var t = 1; t < o.length; t++) {
    p += Math.log(a[x[t - 1]][x[t]]) + Math.log(b[x[t]][o[t]]);
  }

  return Math.exp(p);
};
*/

/*
// This is the solution 1 in chapter 4.1
HiddenMarkovModel.prototype.getProbabilityOfObservations =
function(o, returnLog) {
  this._verifyObservations(o);

  var c = this._alphaPass(o).c;
  var logProb = this._getLogProb(c);

  if (returnLog) {
    return logProb;
  }

  return Math.exp(logProb);
};
*/

/*
// This is the solution 2 in chapter 4.2
HiddenMarkovModel.prototype.getOptimalStateSequencesOfObservations =
function(o) {
  var n = this.numberOfStates;
  var T = o.length;

  this._verifyObservations(o);

  var alphaPassResults = this._alphaPass(o);
  var alpha = alphaPassResults.alpha;
  var c = alphaPassResults.c;
  var beta = this._betaPass(c, o);
  var gamma = this._gammaPass(alpha, beta, c, o).gamma;

  var i, t, maxVal;

  var seq = [];
  for (t = 0; t < T - 1; t++) {
    maxVal = 0;
    for (i = 0; i < n; i++) {
      if (gamma[t * n + i] > maxVal) {
        seq[t] = i;
        maxVal = gamma[t * n + i];
      }
    }
  }

  // The gamma pass from part 4 does not give us gamma[T - 1][i],
  // so we have to compare alpha instead.
  maxVal = 0;
  for (i = 0; i < n; i++) {
    if (alpha[(T - 1) * n + i] > maxVal) {
      seq[t] = i;
      maxVal = alpha[(T - 1) * n + i];
    }
  }

  return seq;
};
*/


HiddenMarkovModel.prototype.simulateStates = function(pathLength, withObservations) {

  var states = [];
  var observations = [];

  for (var t=0; t<pathLength; t++) {
    states.push(-1); // dummy assignment to adjust vector length
    var nextprobas;
    if (t==0) {
      nextprobas = this.initialStateDistributionMatrix[0];
    }
    else
    {
      nextprobas = this.stateTransitionMatrix[states[t-1]];
    }
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

// Train model to fit the observations.
// This method will modify the model matrices.
// This is the solution 3 in chapter 4.3, implemented with code in chapter 7.
HiddenMarkovModel.prototype.fitObservations = function(o, maxIters, verbose) {
  this._verifyObservations(o);

  var oldLogProb = -Infinity;

  for (var iter = 0; iter < maxIters; iter++) {
    var alphaPassResults = this._alphaPass(o);
    var alpha = alphaPassResults.alpha;
    if (verbose)
    {
      console.log(alpha);
    }
    var c = alphaPassResults.c;
    var beta = this._betaPass(c, o);
    if (verbose)
    {
      console.log(beta);
    }
    var gammaPassResults = this._gammaPass(alpha, beta, c, o);
    var gamma = gammaPassResults.gamma;
    var digamma = gammaPassResults.digamma;

    this._updateModel(gamma, digamma, o);

    var logProb = this._getLogProb(c);

    if (logProb <= oldLogProb) {
      break;
    }

    oldLogProb = logProb;
  }

  return iter;
};


/*
// This finds the dynamic programming solution. See chapter 5.
HiddenMarkovModel.prototype.getMostProbableStateSequencesOfObservations =
function(o) {
  var n = this.numberOfStates;
  var a = this.stateTransitionMatrix;
  var b = this.observationProbabilityMatrix;
  var pi = this.initialStateDistributionMatrix;
  var T = o.length;

  this._verifyObservations(o);

  var i, j, t;

  var pathPointers = new Utils.UintSizedArray(n, (T - 1) * n);

  // In the interest of saving memory spaces,
  // we keep only the last delta(t) values instead of the whole series.
  // Keeping the whole series allow us to reproduce Figure 2 though.
  var currentDelta = new Float64Array(n);
  for (i = 0; i < n; i++) {
    currentDelta[i] = Math.log(pi[0][i] * b[i][o[0]]);
  }

  var previousDelta;
  for (t = 1; t < T; t++) {
    previousDelta = currentDelta;
    currentDelta = new Float64Array(n);
    for (i = 0; i < n; i++) {
      currentDelta[i] = -Infinity;
      for (j = 0; j < n; j++) {
        var s =
          previousDelta[j] + Math.log(a[i][j]) + Math.log(b[i][o[t]]);
        if (currentDelta[i] < s) {
          currentDelta[i] = s;
          pathPointers[(t - 1) * n + i] = j;
        }
      }
    }
  }

  var score = -Infinity;
  var stateSequences = new Array(T);

  for (i = 0; i < n; i++) {
    if (score < currentDelta[i]) {
      score = currentDelta[i];
      stateSequences[T - 1] = i;
    }
  }

  t = T - 1;
  var currentState = stateSequences[T - 1];
  while (t--) {
    currentState = stateSequences[t] = pathPointers[t * n + currentState];
  }

  return stateSequences;
};
*/

// This is the 2nd part of chapter 7: The α-pass
HiddenMarkovModel.prototype._alphaPass = function(o) {
  var n = this.numberOfStates;
  var a = this.stateTransitionMatrix;
  var b = this.observationProbabilityCPDs;
  var pi = this.initialStateDistributionMatrix;
  var T = o.length;

  var i, j, t;

  var c = new Float64Array(T);
  var alpha = new Float64Array(T * n);

  c[0] = 0;
  for (i = 0; i < n; i++) {
    alpha[0 * n + i] = pi[0][i] * b[i].pdf(o[0]);
    c[0] += alpha[0 * n + i];
  }

  c[0] = 1 / c[0];
  for (i = 0; i < n; i++) {
    alpha[0 * n + i] *= c[0];
  }

  for (t = 1; t < T; t++) {
    //c[t] = 0;
    for (i = 0; i < n; i++) {
      alpha[t * n + i] = 0;
      for (j = 0; j < n; j++) {
        alpha[t * n + i] +=
          alpha[(t - 1) * n + j] * a[j][i];
      }
      alpha[t * n + i] *= b[i].pdf(o[t]);
      c[t] += alpha[t * n + i];
    }

    c[t] = 1 / c[t];
    for (i = 0; i < n; i++) {
      alpha[t * n + i] *= c[t];
    }
  }

  return {
    c: c,
    alpha: alpha
  };
};

// Part 3 of chapter 7: The β-pass
HiddenMarkovModel.prototype._betaPass = function(c, o) {
  var n = this.numberOfStates;
  var T = o.length;

  var i, j, t;

  var beta = new Float64Array(T * n);

  for (i = 0; i < n; i++) {
    beta[(T - 1) * n + i] = c[(T - 1)];
    //beta[(T - 1) * n + i] = 1;
  }

  for (t = T - 2; t >= 0; t--) {
    for (i = 0; i < n; i++) {
      beta[t * n + i] = 0;
      for (j = 0; j < n; j++) {
        beta[t * n + i] +=
          this.stateTransitionMatrix[i][j] *
          this.observationProbabilityCPDs[j].pdf(o[t + 1]) *
          beta[(t + 1) * n + j];
      }
      beta[t * n + i] *= c[t];
    }
  }

  return beta;
};

// Part 4 of chapter 7: Compute γt(i, j) and γt(i)
HiddenMarkovModel.prototype._gammaPass = function(alpha, beta, c, o) {
  var n = this.numberOfStates;
  //var T = c.length;
  var T = o.length;

  // WARNING
  // should check that 'alpha', 'beta' and 'o' have same t-length
  // what is 'c ? seems useless

  var gamma = new Float64Array(T * n);
  var digamma = new Float64Array(T * n * n);

  // WARNING :
  // 'digamma' ('xi' on Wikipedia for Baum-Welsh) seems correct
  // but 'gamma' itself looks wrong

  var i, j, t;

  for (t = 0; t < T - 1; t++) {
    var denom = 0;
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        denom +=
          alpha[t * n + i] * 
          this.stateTransitionMatrix[i][j] *
          this.observationProbabilityCPDs[j].pdf(o[t + 1]) *
          beta[(t + 1) * n + j];
      }
    }
    for (i = 0; i < n; i++) {
      gamma[t * n + i] = 0;
      for (j = 0; j < n; j++) {
        digamma[t * n * n + i * n + j] =
          (alpha[t * n + i] *
            this.stateTransitionMatrix[i][j] *
            this.observationProbabilityCPDs[j].pdf(o[t + 1]) *
            beta[(t + 1) * n + j]) / denom;
        gamma[t * n + i] += digamma[t * n * n + i * n + j];
      }
    }
  }

  return {
    gamma: gamma,
    digamma: digamma
  };
};

HiddenMarkovModel.prototype._updateModel = function(gamma, digamma, o) {
  var n = this.numberOfStates;
  //var m = this.numberOfObservationSymbols;
  var T = o.length;
  var obsdim = this.dimensionOfObservations;

  var i, j, t, numer, denom;

  for (i = 0; i < n; i++) {
    this.initialStateDistributionMatrix[0][i] = gamma[0 * n + i];
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      numer = 0;
      denom = 0;
      for (t = 0; t < T - 1; t++) {
        numer += digamma[t * n * n + i * n + j];
        denom += gamma[t * n + i];
      }
      this.stateTransitionMatrix[i][j] = numer / denom;
    }
  }

  for (i = 0; i < n; i++) {
    // maximization step : wrt to means
    var newMeans = numeric.mul(this.observationProbabilityCPDs[i].means, 0); // null vector of correct dimension
    denom = 0;
    for (t = 0; t < T - 1; t++) {
      newMeans = numeric.add(newMeans, numeric.mul(gamma[t * n + i], o[t]));
      denom += gamma[t * n + i];
    }
    newMeans = numeric.div(newMeans, denom);

    // maximization step : wrt to covariances
    var newVars = numeric.mul(this.observationProbabilityCPDs[i].covariances, 0); // null matrix of correct dimension
    for (t = 0; t < T - 1; t++) {
      proba = gamma[t * n + i];
      for (j=0; j<obsdim; j++) {
        for (k=0; k<obsdim; k++) {
          newVars[j][k] = newVars[j][k] + proba * (o[t][j]-newMeans[j]) * (o[t][k]-newMeans[k]);
        }; // k
      }; // j
    };
    newVars = numeric.div(newVars, denom);
    this.observationProbabilityCPDs[i] = new GaussianLaw(newMeans, newVars);
  }
};


HiddenMarkovModel.prototype._verifyObservations = function(o) {
  var T = o.length;
  //var m = this.numberOfObservationSymbols;
  var obsdim = this.dimensionOfObservations;

  for (var t = 0; t < T; t++) {
    /*if (typeof o[t] === 'number' && o[t] % 1 === 0 && o[t] >= 0 && o[t] < m) {
      continue;
    }*/
    if (o[t].length === obsdim ) {
      continue;
    }

    throw new Error('HiddenMarkovModel: ' + 'observations at t=' + t + ' is invalid.');
  }
};

// Part 6 of chapter 7: Compute log[P (O | λ)]
HiddenMarkovModel.prototype._getLogProb = function(c) {
  var T = c.length;

  var logProb = 0;
  for (var t = 0; t < T; t++) { // WARNING : WRONG IF IS SET TO C=0
    logProb += Math.log(c[t]);
  }

  return -logProb;
};


//module.exports = HiddenMarkovModel;