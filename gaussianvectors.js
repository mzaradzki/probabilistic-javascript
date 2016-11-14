

function Cholesky(A) {
	// Adapted from this Python version :
	// https://rosettacode.org/wiki/Cholesky_decomposition#Python2.X_version
	if (!(numeric.dim(A)[0] === numeric.dim(A)[1])) {
		throw new Error('Dimension mismatch : '+numeric.dim(A)[0]+' '+numeric.dim(A)[1]);
	}
	var size = numeric.dim(A)[0];
	var L = numeric.mul(A, 0);
	for (var i=0; i<size; i++)
	{
		for (var j=0; j<i+1; j++)
		{
			var s = 0;
			for (var k=0; k<j; k++) {
				s+= L[i][k] * L[j][k];
			}
			if (i == j) {
				L[i][j] = Math.sqrt(A[i][i] - s);
			}
			else {
				L[i][j] = 1.0 / L[j][j] * (A[i][j] - s);
			} 
		}       
	}
	return L;
}

var GaussianLaw = function(means, covariances) {

	if (!(numeric.dim(covariances)[0] === numeric.dim(means)[0])) {
		throw new Error('Dimension mismatch (0) : '+numeric.dim(covariances)[0]+' '+numeric.dim(means)[0]);
	}
	if (!(numeric.dim(covariances)[1] === numeric.dim(means)[0])) {
		throw new Error('Dimension mismatch (1) : '+numeric.dim(covariances)[1]+' '+numeric.dim(means)[0]);
	}

	this.dimension = numeric.dim(means);
	this.means = means;
	this.covariances = covariances;
	this.precisions = numeric.inv(covariances);
	this.cholesky = Cholesky(covariances);
	this.determinant = numeric.det(covariances);
	this.determinantsqrt = Math.pow(this.determinant, -0.5);
	
	this.determinantlogsqrt = -0.5 * Math.log(this.determinant);
	this.pdfscale = Math.pow(2*Math.PI, -0.5*this.dimension);
	this.pdfscalelog = -0.5 * this.dimension * Math.log(2*Math.PI);
};

GaussianLaw.prototype.pdf = function(X) {
	var dA = numeric.dotVV(numeric.sub(X,this.means), numeric.dot(this.precisions, numeric.sub(X,this.means)));
	return this.pdfscale * this.determinantsqrt * Math.exp(-0.5*dA);
};

GaussianLaw.prototype.logpdf = function(X) {
	var dA = numeric.dotVV(numeric.sub(X,this.means), numeric.dot(this.precisions, numeric.sub(X,this.means)));
	return this.pdfscalelog + this.determinantlogsqrt + (-0.5*dA);
};

GaussianLaw.prototype.simulate = function() {
	var stdvec = [];
	for (var i=0; i<this.dimension; i++) {
		// Box Muller method
		var u = 1 - Math.random(); // Subtraction to flip [0, 1) to (0, 1].
		var v = 1 - Math.random();
		var g = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
		stdvec.push(g);
	}
	var covvec = numeric.dot(this.cholesky, stdvec);
	return numeric.add(this.means, covvec);
};

