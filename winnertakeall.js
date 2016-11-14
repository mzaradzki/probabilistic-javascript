

function WinnerTakeAll(observations, nbclasses, lambda, nbepochs) {
	//var lambda = 0.01;
	if (observations.length<nbclasses) {
		throw new Error('Too many classes for so few observations');
	}
	guesses = [];
	for (var c=0; c<nbclasses; c++) {
		guesses.push(observations[c]); // TO DO : here we could randomize
	}
	for(var i=0; i<nbepochs; i++) {
		var point = observations[Math.floor(Math.random()*observations.length)];
	  var bestclass;
	  var bestdelta;
	  var bestnorm2 = -1;
	  for (var c=0; c<nbclasses; c++) {
		var delta = numeric.sub(point, guesses[c]);
		var norm2 = numeric.dot(delta, delta);
		if ((c==0) || (norm2<bestnorm2)) {
			bestclass = c;
			bestdelta = delta;
			bestnorm2 = norm2;
		}
	  }
	  guesses[bestclass] = numeric.add(guesses[bestclass], numeric.mul(bestdelta, lambda));
	}
	return guesses;
}