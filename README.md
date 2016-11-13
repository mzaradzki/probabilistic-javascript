# probabilistic-javascript

Javascript functions to fit and to simulate probabilistic models


### Gaussian Vectors

    var means = [0.5, 0.5, 0.5];
    var covariance = [[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]];
    var glaw = new GaussianLaw(means, covariances);
    console.log(glaw.simulate());
    console.log(glaw.simulate());

### Gaussian Mixture Model (multi-dimensional)
Model estimation using EM (expectation-maximization) algorithm

    var nbstates = 2;
    var obsdim = 3;
    var gmm = new GaussianMixtureModel(nbstates, obsdim);

### Hidden Markov Model (multi-dimensional)
Model estimation using Baum-Welch algorithm

    var nbstates = 2;
    var obsdim = 3;
    var hmm1 = new HiddenMarkovModel(nbstates, obsdim);
    var pathLength = 200;
    var points = hmm1.simulateStates(pathLength, true).observations;
    //
    var hmm2 = new HiddenMarkovModel(nbstates, obsdim);
    var maxIters = 50;
    hmm2.fitObservations(points, maxIters, false);
    //
    console.log(hmm2.stateTransitionMatrix);
    console.log(hmm2.observationProbabilityCPDs[0].means);
    console.log(hmm2.observationProbabilityCPDs[1].means);
    console.log(hmm2.observationProbabilityCPDs[2].means);


### Dependencies
Rely on package numeric.js for matrix algebra

    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/numeric/1.2.6/numeric.js"></script>

