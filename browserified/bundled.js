(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){

var dnn = require('dnn'); // see documentation : https://www.npmjs.com/package/dnn
},{"dnn":9}],2:[function(require,module,exports){
/**
 * Created by joonkukang on 2014. 1. 13..
 */
var math = require('./utils').math;
LogisticRegression = require('./LogisticRegression');
HiddenLayer = require('./HiddenLayer');
RBM = require('./RBM');
CRBM = require('./CRBM');
DBN = require('./DBN');


CDBN = module.exports = function (settings) {
    var self = this;
    self.x = settings['input'];
    self.y = settings['label'];
    self.sigmoidLayers = [];
    self.rbmLayers = [];
    self.nLayers = settings['hidden_layer_sizes'].length;
    self.hiddenLayerSizes = settings['hidden_layer_sizes'];
    self.nIns = settings['n_ins'];
    self.nOuts = settings['n_outs'];

    self.settings = {
        'log level' : 1 // 0 : nothing, 1 : info, 2: warn
    };
    // Constructing Deep Neural Network
    var i;
    for(i=0 ; i<self.nLayers ; i++) {
        var inputSize, layerInput;
        if(i == 0)
            inputSize = settings['n_ins'];
        else
            inputSize = settings['hidden_layer_sizes'][i-1];

        if(i == 0)
            layerInput = self.x;
        else
            layerInput = self.sigmoidLayers[self.sigmoidLayers.length-1].sampleHgivenV();

        var sigmoidLayer = new HiddenLayer({
            'input' : layerInput,
            'n_in' : inputSize,
            'n_out' : settings['hidden_layer_sizes'][i],
            'activation' : math.sigmoid
        });
        self.sigmoidLayers.push(sigmoidLayer);

        var rbmLayer;
        if(i==0) {
            rbmLayer = new CRBM({
                'input' : layerInput,
                'n_visible' : inputSize,
                'n_hidden' : settings['hidden_layer_sizes'][i],
            });
        } else {
            rbmLayer = new RBM({
                'input' : layerInput,
                'n_visible' : inputSize,
                'n_hidden' : settings['hidden_layer_sizes'][i]
            });
        }
        self.rbmLayers.push(rbmLayer);
    }
    self.outputLayer = new HiddenLayer({
        'input' : self.sigmoidLayers[self.sigmoidLayers.length-1].sampleHgivenV(),
        'n_in' : settings['hidden_layer_sizes'][settings['hidden_layer_sizes'].length - 1],
        'n_out' : settings['n_outs'],
        'activation' : math.sigmoid
    });
};

CDBN.prototype.__proto__ = DBN.prototype;
},{"./CRBM":3,"./DBN":4,"./HiddenLayer":5,"./LogisticRegression":6,"./RBM":8,"./utils":11}],3:[function(require,module,exports){
/**
 * Created by joonkukang on 2014. 1. 13..
 */

var math = require('./utils').math;
RBM = require('./RBM');

CRBM = module.exports = function (settings) {
    RBM.call(this,settings);
};

CRBM.prototype = new RBM({});

CRBM.prototype.propdown = function(h) {
    var self = this;
    var preSigmoidActivation = math.addMatVec(math.mulMat(h,math.transpose(self.W)),self.vbias);
    return preSigmoidActivation;
};

CRBM.prototype.sampleVgivenH = function(h0_sample) {
    var self = this;
    var a_h = self.propdown(h0_sample);
    var a = math.activateMat(a_h,function(x) { return 1. / (1-Math.exp(-x)) ; });
    var b = math.activateMat(a_h,function(x){ return 1./x ;});
    var v1_mean = math.minusMat(a,b);
    var U = math.randMat(math.shape(v1_mean)[0],math.shape(v1_mean)[1],0,1);
    var c = math.activateMat(a_h,function(x) { return 1 - Math.exp(x);});
    var d = math.activateMat(math.mulMatElementWise(U,c),function(x) {return 1-x;});
    var v1_sample = math.activateTwoMat(math.activateMat(d,Math.log),a_h,function(x,y) {
        if(y==0) y += 1e-14; // Javascript Float Precision Problem.. This is a limit of javascript.
        return x/y;
    })
    return [v1_mean,v1_sample];
};
CRBM.prototype.getReconstructionCrossEntropy = function() {
    var self = this;
    var reconstructedV = self.reconstruct(self.input);
    var a = math.activateTwoMat(self.input,reconstructedV,function(x,y){
        return x*Math.log(y);
    });

    var b = math.activateTwoMat(self.input,reconstructedV,function(x,y){
        return (1-x)*Math.log(1-y);
    });

    var crossEntropy = -math.meanVec(math.sumMatAxis(math.addMat(a,b),1));
    return crossEntropy;

};

CRBM.prototype.reconstruct = function(v) {
    var self = this;
    var reconstructedV = self.sampleVgivenH(self.sampleHgivenV(v)[0])[0];
    return reconstructedV;
};
},{"./RBM":8,"./utils":11}],4:[function(require,module,exports){
/**
 * Created by joonkukang on 2014. 1. 13..
 */
var math = require('./utils').math;
HiddenLayer = require('./HiddenLayer');
RBM = require('./RBM');
MLP = require('./MLP');

DBN = module.exports = function (settings) {
    var self = this;
    self.x = settings['input'];
    self.y = settings['label'];
    self.sigmoidLayers = [];
    self.rbmLayers = [];
    self.nLayers = settings['hidden_layer_sizes'].length;
    self.hiddenLayerSizes = settings['hidden_layer_sizes'];
    self.nIns = settings['n_ins'];
    self.nOuts = settings['n_outs'];
    self.settings = {
        'log level' : 1 // 0 : nothing, 1 : info, 2: warn
    };

    // Constructing Deep Neural Network
    var i;
    for(i=0 ; i<self.nLayers ; i++) {
        var inputSize, layerInput;
        if(i == 0)
            inputSize = settings['n_ins'];
        else
            inputSize = settings['hidden_layer_sizes'][i-1];

        if(i == 0)
            layerInput = self.x;
        else
            layerInput = self.sigmoidLayers[self.sigmoidLayers.length-1].sampleHgivenV();

        var sigmoidLayer = new HiddenLayer({
            'input' : layerInput,
            'n_in' : inputSize,
            'n_out' : settings['hidden_layer_sizes'][i],
            'activation' : math.sigmoid
        });
        self.sigmoidLayers.push(sigmoidLayer);

        var rbmLayer = new RBM({
            'input' : layerInput,
            'n_visible' : inputSize,
            'n_hidden' : settings['hidden_layer_sizes'][i]
        });
        self.rbmLayers.push(rbmLayer);
    }
    self.outputLayer = new HiddenLayer({
        'input' : self.sigmoidLayers[self.sigmoidLayers.length-1].sampleHgivenV(),
        'n_in' : settings['hidden_layer_sizes'][settings['hidden_layer_sizes'].length - 1],
        'n_out' : settings['n_outs'],
        'activation' : math.sigmoid
    });
};

DBN.prototype.pretrain = function (settings) {
    var self = this;
    var lr = 0.6, k = 1, epochs = 2000;
    if(typeof settings['lr'] !== 'undefined')
        lr = settings['lr'];
    if(typeof settings['k'] !== 'undefined')
        k = settings['k'];
    if(typeof settings['epochs'] !== 'undefined')
        epochs = settings['epochs'];

    var i,j;
    for(i=0; i<self.nLayers ; i++) {
        var layerInput ,rbm;
        if (i==0)
            layerInput = self.x;
        else
            layerInput = self.sigmoidLayers[i-1].sampleHgivenV(layerInput);
        rbm = self.rbmLayers[i];
        rbm.set('log level',0);
        rbm.train({
            'lr' : lr,
            'k' : k,
            'input' : layerInput,
            'epochs' : epochs
        });

        if(self.settings['log level'] > 0) {
            console.log("DBN RBM",i,"th Layer Final Cross Entropy: ",rbm.getReconstructionCrossEntropy());
            console.log("DBN RBM",i,"th Layer Pre-Training Completed.");
        }

        // Synchronization between RBM and sigmoid Layer
        self.sigmoidLayers[i].W = rbm.W;
        self.sigmoidLayers[i].b = rbm.hbias;
    }
    if(self.settings['log level'] > 0)
        console.log("DBN Pre-Training Completed.")
};

DBN.prototype.finetune = function (settings) {
    var self = this;
    var lr = 0.2, epochs = 1000;
    if(typeof settings['lr'] !== 'undefined')
        lr = settings['lr'];
    if(typeof settings['epochs'] !== 'undefined')
        epochs = settings['epochs'];

    //Fine-Tuning Using MLP (Back Propagation)
    var i;
    var pretrainedWArray = [], pretrainedBArray = []; // HiddenLayer W,b values already pretrained by RBM.
    for(i=0; i<self.nLayers ; i++) {
        pretrainedWArray.push(self.sigmoidLayers[i].W);
        pretrainedBArray.push(self.sigmoidLayers[i].b);
    }
    // W,b of Final Output Layer are not involved in pretrainedWArray, pretrainedBArray so they will be treated as undefined at MLP Constructor.
    var mlp = new MLP({
        'input' : self.x,
        'label' : self.y,
        'n_ins' : self.nIns,
        'n_outs' : self.nOuts,
        'hidden_layer_sizes' : self.hiddenLayerSizes,
        'w_array' : pretrainedWArray,
        'b_array' : pretrainedBArray
    });
    mlp.set('log level',self.settings['log level']);
    mlp.train({
        'lr' : lr,
        'epochs' : epochs
    });
    for(i=0; i<self.nLayers ; i++) {
        self.sigmoidLayers[i].W = mlp.sigmoidLayers[i].W;
        self.sigmoidLayers[i].b = mlp.sigmoidLayers[i].b;
    }
    self.outputLayer.W = mlp.sigmoidLayers[self.nLayers].W;
    self.outputLayer.b = mlp.sigmoidLayers[self.nLayers].b;

};

DBN.prototype.getReconstructionCrossEntropy = function() {
    var self = this;
    var reconstructedOutput = self.predict(self.x);
    var a = math.activateTwoMat(self.y,reconstructedOutput,function(x,y){
        return x*Math.log(y);
    });

    var b = math.activateTwoMat(self.y,reconstructedOutput,function(x,y){
        return (1-x)*Math.log(1-y);
    });

    var crossEntropy = -math.meanVec(math.sumMatAxis(math.addMat(a,b),1));
    return crossEntropy
};

DBN.prototype.predict = function (x) {
    var self = this;
    var layerInput = x, i;
    for(i=0; i<self.nLayers ; i++) {
        layerInput = self.sigmoidLayers[i].output(layerInput);
    }
    var output = self.outputLayer.output(layerInput);
    return output;
};

DBN.prototype.set = function(property,value) {
    var self = this;
    self.settings[property] = value;
}
},{"./HiddenLayer":5,"./MLP":7,"./RBM":8,"./utils":11}],5:[function(require,module,exports){
/**
 * Created by joonkukang on 2014. 1. 12..
 */
var math = require('./utils').math;
HiddenLayer = module.exports = function (settings) {
    var self = this;
    self.input = settings['input'];

    if(typeof settings['W'] === 'undefined') {
        var a = 1. / settings['n_in'];
        settings['W'] = math.randMat(settings['n_in'],settings['n_out'],-a,a);
    }
    if(typeof settings['b'] === 'undefined')
        settings['b'] = math.zeroVec(settings['n_out']);
    if(typeof settings['activation'] === 'undefined')
        settings['activation'] = math.sigmoid;

    self.W = settings['W'];
    self.b = settings['b'];
    self.activation = settings['activation'];
}

HiddenLayer.prototype.output = function(input) {
    var self = this;
    if(typeof input !== 'undefined')
        self.input = input;

    var linearOutput = math.addMatVec(math.mulMat(self.input,self.W),self.b);
    return math.activateMat(linearOutput,self.activation);
};

HiddenLayer.prototype.linearOutput = function(input) { // returns the value before activation.
    var self = this;
    if(typeof input !== 'undefined')
        self.input = input;

    var linearOutput = math.addMatVec(math.mulMat(self.input,self.W),self.b);
    return linearOutput;
}

HiddenLayer.prototype.backPropagate = function (input) { // example+num * n_out matrix
    var self = this;
    if(typeof input === 'undefined')
        throw new Error("No BackPropagation Input.")

    var linearOutput = math.mulMat(input, m.transpose(self.W));
    return linearOutput;
}

HiddenLayer.prototype.sampleHgivenV = function(input) {
    var self = this;
    if(typeof input !== 'undefined')
        self.input = input;

    var hMean = self.output();
    var hSample = math.probToBinaryMat(hMean);
    return hSample;
}
},{"./utils":11}],6:[function(require,module,exports){
/**
 * Created by joonkukang on 2014. 1. 12..
 */
var math = require('./utils').math;
LogisticRegression = module.exports = function (settings) {
    var self = this;
    self.x = settings['input'];
    self.y = settings['label'];
    self.W = math.zeroMat(settings['n_in'],settings['n_out']);
    self.b = math.zeroVec(settings['n_out']);
    self.settings = {
        'log level' : 1 // 0 : nothing, 1 : info, 2: warn
    };
};

LogisticRegression.prototype.train = function (settings) {
    var self = this;
    var lr = 0.1, epochs = 200;
    if(typeof settings['input'] !== 'undefined')
        self.x = settings['input'];
    if(typeof settings['lr'] !== 'undefined')
        lr = settings['lr'];
    if(typeof settings['epochs'] !== 'undefined')
        epochs = settings['epochs'];
    var i;
    var currentProgress = 1;
    for(i=0;i<epochs;i++) {
        var probYgivenX = math.softmaxMat(math.addMatVec(math.mulMat(self.x,self.W),self.b));
        var deltaY = math.minusMat(self.y,probYgivenX);

        var deltaW = math.mulMat(math.transpose(self.x),deltaY);
        var deltaB = math.meanMatAxis(deltaY,0);

        self.W = math.addMat(self.W,math.mulMatScalar(deltaW,lr));
        self.b = math.addVec(self.b,math.mulVecScalar(deltaB,lr));
        if(self.settings['log level'] > 0) {
            var progress = (1.*i/epochs)*100;
            if(progress > currentProgress) {
                console.log("LogisticRegression",progress.toFixed(0),"% Completed.");
                currentProgress++;
            }
        }
    }
    if(self.settings['log level'] > 0)
        console.log("LogisticRegression Final Cross Entropy : ",self.getReconstructionCrossEntropy());
};

LogisticRegression.prototype.getReconstructionCrossEntropy = function () {
    var self = this;
    var probYgivenX = math.softmaxMat(math.addMatVec(math.mulMat(self.x,self.W),self.b));
    var a = math.mulMatElementWise(self.y, math.activateMat(probYgivenX,Math.log));
    var b = math.mulMatElementWise(math.mulMatScalar(math.addMatScalar(self.y,-1),-1),
        math.activateMat(math.mulMatScalar(math.addMatScalar(probYgivenX,-1),-1),Math.log));
    var crossEntropy = -math.meanVec(math.sumMatAxis(math.addMat(a,b),1));
    return crossEntropy;
};

LogisticRegression.prototype.predict = function (x) {
    var self = this;
    return math.softmaxMat(math.addMatVec(math.mulMat(x,self.W),self.b));
};

LogisticRegression.prototype.set = function(property,value) {
    var self = this;
    self.settings[property] = value;
}
},{"./utils":11}],7:[function(require,module,exports){
/**
 * Created by joonkukang on 2014. 1. 14..
 */
var math = require('./utils').math;

MLP = module.exports = function (settings) {
    var self = this;
    self.x = settings['input'];
    self.y = settings['label'];
    self.sigmoidLayers = [];
    self.nLayers = settings['hidden_layer_sizes'].length;
    self.settings = {
        'log level' : 1 // 0 : nothing, 1 : info, 2: warn
    };
    var i;
    for(i=0 ; i<self.nLayers+1 ; i++) {
        var inputSize, layerInput;
        if(i == 0)
            inputSize = settings['n_ins'];
        else
            inputSize = settings['hidden_layer_sizes'][i-1];

        if(i == 0)
            layerInput = self.x;
        else
            layerInput = self.sigmoidLayers[self.sigmoidLayers.length-1].sampleHgivenV();

        var sigmoidLayer;
        if(i == self.nLayers) {
            sigmoidLayer = new HiddenLayer({
                'input' : layerInput,
                'n_in' : inputSize,
                'n_out' : settings['n_outs'],
                'activation' : math.sigmoid,
                'W' : (typeof settings['w_array'] === 'undefined')? undefined : settings['w_array'][i],
                'b' : (typeof settings['b_array'] === 'undefined')? undefined : settings['b_array'][i]
            });
        } else {
            sigmoidLayer = new HiddenLayer({
                'input' : layerInput,
                'n_in' : inputSize,
                'n_out' : settings['hidden_layer_sizes'][i],
                'activation' : math.sigmoid,
                'W' : (typeof settings['w_array'] === 'undefined')? undefined : settings['w_array'][i],
                'b' : (typeof settings['b_array'] === 'undefined')? undefined : settings['b_array'][i]
            });
        }
        self.sigmoidLayers.push(sigmoidLayer);
    }
};

MLP.prototype.train = function(settings) {
    var self = this;
    var lr = 0.6, epochs = 1000;
    if(typeof settings['lr'] !== 'undefined')
        lr = settings['lr'];
    if(typeof settings['epochs'] !== 'undefined')
        epochs = settings['epochs'];


    var epoch;
    var currentProgress = 1;
    for(epoch=0 ; epoch < epochs ; epoch++) {

        // Feed Forward
        var i;
        var layerInput = [];
        layerInput.push(self.x);
        for(i=0; i<self.nLayers+1 ; i++) {
            layerInput.push(self.sigmoidLayers[i].output(layerInput[i]));
        }
        var output = layerInput[self.nLayers+1];
        // Back Propagation
        var delta = new Array(self.nLayers + 1);
        delta[self.nLayers] = m.mulMatElementWise(m.minusMat(self.y, output),
            m.activateMat(self.sigmoidLayers[self.nLayers].linearOutput(layerInput[self.nLayers]), m.dSigmoid));

        /*
         self.nLayers = 3 (3 hidden layers)
         delta[3] : ouput layer
         delta[2] : 3rd hidden layer, delta[0] : 1st hidden layer
         */
        for(i = self.nLayers - 1; i>=0 ; i--) {
            delta[i] = m.mulMatElementWise(self.sigmoidLayers[i+1].backPropagate(delta[i+1]),
                m.activateMat(self.sigmoidLayers[i].linearOutput(layerInput[i]), m.dSigmoid));
        }
        // Update Weight, Bias
        for(var i=0; i<self.nLayers+1 ; i++) {
            var deltaW = m.activateMat(m.mulMat(m.transpose(layerInput[i]),delta[i]),function(x){return 1. * x / self.x.length;})
            var deltaB = m.meanMatAxis(delta[i],0);
            self.sigmoidLayers[i].W = m.addMat(self.sigmoidLayers[i].W,deltaW);
            self.sigmoidLayers[i].b = m.addVec(self.sigmoidLayers[i].b,deltaB);
        }

        if(self.settings['log level'] > 0) {
            var progress = (1.*epoch/epochs)*100;
            if(progress > currentProgress) {
                console.log("MLP",progress.toFixed(0),"% Completed.");
                currentProgress+=8;
            }
        }
    }
    if(self.settings['log level'] > 0)
        console.log("MLP Final Cross Entropy : ",self.getReconstructionCrossEntropy());
};

MLP.prototype.getReconstructionCrossEntropy = function() {
    var self = this;
    var reconstructedOutput = self.predict(self.x);
    var a = math.activateTwoMat(self.y,reconstructedOutput,function(x,y){
        return x*Math.log(y);
    });

    var b = math.activateTwoMat(self.y,reconstructedOutput,function(x,y){
        return (1-x)*Math.log(1-y);
    });

    var crossEntropy = -math.meanVec(math.sumMatAxis(math.addMat(a,b),1));
    return crossEntropy
}

MLP.prototype.predict = function(x) {
    var self = this;
    var output = x;
    for(i=0; i<self.nLayers+1 ; i++) {
        output = self.sigmoidLayers[i].output(output);
    }
    return output;
};

MLP.prototype.set = function(property,value) {
    var self = this;
    self.settings[property] = value;
}
},{"./utils":11}],8:[function(require,module,exports){
var math = require('./utils').math;
RBM = module.exports = function (settings) {
    var self = this;

    self.nVisible = settings['n_visible'];
    self.nHidden = settings['n_hidden'];
    self.settings = {
        'log level' : 1 // 0 : nothing, 1 : info, 2: warn
    };

    if(typeof settings['W'] === 'undefined') {
        var a = 1. / self.nVisible;
        settings['W'] = math.randMat(self.nVisible,self.nHidden,-a,a);
    }

    if(typeof settings['hbias'] === 'undefined')
        settings['hbias'] = math.zeroVec(self.nHidden);

    if(typeof settings['vbias'] === 'undefined')
        settings['vbias'] = math.zeroVec(self.nVisible);

    self.input = settings['input'];
    self.W = settings['W'];
    self.hbias = settings['hbias'];
    self.vbias = settings['vbias'];
}

RBM.prototype.train = function(settings) {
    var self = this;
    var lr=0.8, k= 1, epochs = 1500; // default
    if(typeof settings['input'] !== 'undefined')
        self.input = settings['input'];
    if(typeof settings['lr'] !== 'undefined')
        lr = settings['lr'];
    if(typeof settings['k'] !== 'undefined')
        k = settings['k'];
    if(typeof settings['epochs'] !== 'undefined')
        epochs = settings['epochs'];

    var i,j;
    var currentProgress = 1;
    for(i=0;i<epochs;i++) {
        /* CD - k . Contrastive Divergence */
        var ph = self.sampleHgivenV(self.input);
        var phMean = ph[0], phSample = ph[1];
        var chainStart = phSample;
        var nvMeans, nvSamples, nhMeans, nhSamples;

        for(j=0 ; j<k ; j++) {
            if (j==0) {
                var gibbsVH = self.gibbsHVH(chainStart);
                nvMeans = gibbsVH[0], nvSamples = gibbsVH[1], nhMeans = gibbsVH[2], nhSamples = gibbsVH[3];
            } else {
                var gibbsVH = self.gibbsHVH(nhSamples);
                nvMeans = gibbsVH[0], nvSamples = gibbsVH[1], nhMeans = gibbsVH[2], nhSamples = gibbsVH[3];
            }
        }

        var deltaW = math.mulMatScalar(math.minusMat(math.mulMat(math.transpose(self.input),phMean), math.mulMat(math.transpose(nvSamples),nhMeans)),1. / self.input.length);
        var deltaVbias = math.meanMatAxis(math.minusMat(self.input,nvSamples),0);
        var deltaHbias = math.meanMatAxis(math.minusMat(phSample,nhMeans),0);

        self.W = math.addMat(self.W, math.mulMatScalar(deltaW,lr));
        self.vbias = math.addVec(self.vbias, math.mulVecScalar(deltaVbias,lr));
        self.hbias = math.addVec(self.hbias, math.mulVecScalar(deltaHbias,lr));
        if(self.settings['log level'] > 0) {
            var progress = (1.*i/epochs)*100;
            if(progress > currentProgress) {
                console.log("RBM",progress.toFixed(0),"% Completed.");
                currentProgress+=8;
            }
        }
    }
    if(self.settings['log level'] > 0)
        console.log("RBM Final Cross Entropy : ",self.getReconstructionCrossEntropy())
};

RBM.prototype.propup = function(v) {
    var self = this;
    var preSigmoidActivation = math.addMatVec(math.mulMat(v,self.W),self.hbias);
    return math.activateMat(preSigmoidActivation, m.sigmoid);
};

RBM.prototype.propdown = function(h) {
    var self = this;
    var preSigmoidActivation = math.addMatVec(math.mulMat(h,math.transpose(self.W)),self.vbias);
    return math.activateMat(preSigmoidActivation, m.sigmoid);
};

RBM.prototype.sampleHgivenV = function(v0_sample) {
    var self = this;
    var h1_mean = self.propup(v0_sample);
    var h1_sample = math.probToBinaryMat(h1_mean);
    return [h1_mean,h1_sample];
};

RBM.prototype.sampleVgivenH = function(h0_sample) {
    var self = this;
    var v1_mean = self.propdown(h0_sample);
    var v1_sample = math.probToBinaryMat(v1_mean);
    return [v1_mean,v1_sample];
};

RBM.prototype.gibbsHVH = function(h0_sample) {
    var self = this;
    var v1 = self.sampleVgivenH(h0_sample);
    var h1 = self.sampleHgivenV(v1[1]);
    return [v1[0],v1[1],h1[0],h1[1]];
};

RBM.prototype.reconstruct = function(v) {
    var self = this;
    var h = math.activateMat(math.addMatVec(math.mulMat(v,self.W),self.hbias), math.sigmoid);
    var reconstructedV = math.activateMat(math.addMatVec(math.mulMat(h,math.transpose(self.W)),self.vbias), math.sigmoid);
    return reconstructedV;
};

RBM.prototype.getReconstructionCrossEntropy = function() {
    var self = this;
    var reconstructedV = self.reconstruct(self.input);
    var a = math.activateTwoMat(self.input,reconstructedV,function(x,y){
        return x*Math.log(y);
    });

    var b = math.activateTwoMat(self.input,reconstructedV,function(x,y){
        return (1-x)*Math.log(1-y);
    });

    var crossEntropy = -math.meanVec(math.sumMatAxis(math.addMat(a,b),1));
    return crossEntropy

};
RBM.prototype.set = function(property,value) {
    var self = this;
    self.settings[property] = value;
}
},{"./utils":11}],9:[function(require,module,exports){
/**
 * Created by joonkukang on 2014. 1. 12..
 */
dnn = module.exports;

dnn.RBM = require('./RBM');

dnn.LogisticRegression = require('./LogisticRegression');

dnn.DBN = require('./DBN');

dnn.CRBM = require('./CRBM');

dnn.CDBN = require('./CDBN');

dnn.MLP = require('./MLP');
},{"./CDBN":2,"./CRBM":3,"./DBN":4,"./LogisticRegression":6,"./MLP":7,"./RBM":8}],10:[function(require,module,exports){
/**
 * Created by joonkukang on 2014. 1. 12..
 */
m = module.exports;

m.randn = function() {
    // generate random guassian distribution number. (mean : 0, standard deviation : 1)
    var v1, v2, s;

    do {
        v1 = 2 * Math.random() - 1;   // -1.0 ~ 1.0 까지의 값
        v2 = 2 * Math.random() - 1;   // -1.0 ~ 1.0 까지의 값
        s = v1 * v1 + v2 * v2;
    } while (s >= 1 || s == 0);

    s = Math.sqrt( (-2 * Math.log(s)) / s );
    return v1 * s;
}

m.shape = function(mat) {
    var row = mat.length;
    var col = mat[0].length;
    return [row,col];
};

m.addVec = function(vec1, vec2) {
    if(vec1.length === vec2.length) {
        var result = [];
        var i;
        for(i=0;i<vec1.length;i++)
            result.push(vec1[i]+vec2[i]);
        return result;
    } else {
        throw new Error("Length Error : not same.")
    }
}

m.minusVec = function(vec1,vec2) {
    if(vec1.length === vec2.length) {
        var result = [];
        var i;
        for(i=0;i<vec1.length;i++)
            result.push(vec1[i]-vec2[i]);
        return result;
    } else {
        throw new Error("Length Error : not same.")
    }
};

m.addMatScalar = function(mat,scalar) {
    var row = m.shape(mat)[0];
    var col = m.shape(mat)[1];
    var i , j,result = [];
    for(i=0 ; i<row ; i++) {
        var rowVec = [];
        for(j=0 ; j<col ; j++) {
            rowVec.push(mat[i][j] + scalar);
        }
        result.push(rowVec);
    }
    return result;
}

m.addMatVec = function(mat,vec) {
    if(mat[0].length === vec.length) {
        var result = [];
        var i;
        for(i=0;i<mat.length;i++)
            result.push(m.addVec(mat[i],vec));
        return result;
    } else {
        throw new Error("Length Error : not same.")
    }
}

m.minusMatVec = function(mat,vec) {
    if(mat[0].length === vec.length) {
        var result = [];
        var i;
        for(i=0;i<mat.length;i++)
            result.push(m.minusVec(mat[i],vec));
        return result;
    } else {
        throw new Error("Length Error : not same.")
    }
}

m.addMat = function (mat1, mat2) {
    if ((mat1.length === mat2.length) && (mat1[0].length === mat2[0].length)) {
        var result = new Array(mat1.length);
        for (var i = 0; i < mat1.length; i++) {
            result[i] = new Array(mat1[i].length);
            for (var j = 0; j < mat1[i].length; j++) {
                result[i][j] = mat1[i][j] + mat2[i][j];
            }
        }
        return result;
    } else {
        throw new Error('Matrix mismatch.');
    }
};

m.minusMat = function(mat1, mat2) {
    if ((mat1.length === mat2.length) && (mat1[0].length === mat2[0].length)) {
        var result = new Array(mat1.length);
        for (var i = 0; i < mat1.length; i++) {
            result[i] = new Array(mat1[i].length);
            for (var j = 0; j < mat1[i].length; j++) {
                result[i][j] = mat1[i][j] - mat2[i][j];
            }
        }
        return result;
    } else {
        throw new Error('Matrix mismatch.');
    }
}

m.transpose = function (mat) {
    var result = new Array(mat[0].length);
    for (var i = 0; i < mat[0].length; i++) {
        result[i] = new Array(mat.length);
        for (var j = 0; j < mat.length; j++) {
            result[i][j] = mat[j][i];
        }
    }
    return result;
};

m.dotVec = function (vec1, vec2) {
    if (vec1.length === vec2.length) {
        var result = 0;
        for (var i = 0; i < vec1.length; i++) {
            result += vec1[i] * vec2[i];
        }
        return result;
    } else {
        throw new Error("Vector mismatch");
    }
};

m.outerVec = function (vec1,vec2) {
    var mat1 = m.transpose([vec1]);
    var mat2 = [vec2];
    return m.mulMat(mat1,mat2);
};

m.mulVecScalar = function(vec,scalar) {
    var i, result = [];
    for(i=0;i<vec.length;i++)
        result.push(vec[i]*scalar);
    return result;
};

m.mulMatScalar = function(mat,scalar) {
    var row = m.shape(mat)[0];
    var col = m.shape(mat)[1];
    var i , j,result = [];
    for(i=0 ; i<row ; i++) {
        var rowVec = [];
        for(j=0 ; j<col ; j++) {
            rowVec.push(mat[i][j] * scalar);
        }
        result.push(rowVec);
    }
    return result;
};

m.mulMatElementWise = function(mat1, mat2) {
    if (mat1.length === mat2.length && mat1[0].length === mat2[0].length) {
        var result = new Array(mat1.length);

        for (var x = 0; x < mat1.length; x++) {
            result[x] = new Array(mat1[0].length);
        }

        for (var i = 0; i < result.length; i++) {
            for (var j = 0; j < result[i].length; j++) {
                result[i][j] = mat1[i][j] * mat2[i][j]
            }
        }
        return result;
    } else {
        throw new Error("Matrix shape error : not same");
    }
};

m.mulMat = function (mat1, mat2) {
    if (mat1[0].length === mat2.length) {
        var result = new Array(mat1.length);

        for (var x = 0; x < mat1.length; x++) {
            result[x] = new Array(mat2[0].length);
        }


        var mat2_T = m.transpose(mat2);
        for (var i = 0; i < result.length; i++) {
            for (var j = 0; j < result[i].length; j++) {
                result[i][j] = m.dotVec(mat1[i],mat2_T[j]);
            }
        }
        return result;
    } else {
        throw new Error("Array mismatch");
    }
};

m.sumVec = function(vec) {
    var sum = 0;
    var i = vec.length;
    while (i--) {
        sum += vec[i];
    }
    return sum;
};

m.sumMat = function(mat) {
    var sum = 0;
    var i = mat.length;
    while (i--) {
        for(var j=0;j<mat[0].length;j++)
          sum += mat[i][j];
    }
    return sum;
};

m.sumMatAxis = function(mat,axis) {
    // default axis 0;
    // axis 0 : mean of col vector . axis 1 : mean of row vector
    if(axis === 1) {
        var row = m.shape(mat)[0];
        var i ;
        var result = [];
        for(i=0 ; i<row; i++)
            result.push(m.sumVec(mat[i]));
        return result;
    } else {
        mat_T = m.transpose(mat);
        return m.sumMatAxis(mat_T,1);
    }
};

m.meanVec = function(vec) {
    return 1. * m.sumVec(vec) / vec.length;
};

m.meanMat = function(mat) {
    var row = mat.length;
    var col = mat[0].length;
    return 1. * m.sumMat(mat) / (row * col);
};

m.meanMatAxis = function(mat,axis) {
    // default axis 0;
    // axis 0 : mean of col vector . axis 1 : mean of row vector
    if(axis === 1) {
        var row = m.shape(mat)[0];
        var i ;
        var result = [];
        for(i=0 ; i<row; i++)
            result.push(m.meanVec(mat[i]));
        return result;
    } else {
        mat_T = m.transpose(mat);
        return m.meanMatAxis(mat_T,1);
    }
};

m.squareVec = function(vec) {
    var squareVec = [];
    var i;
    for(i=0;i<vec.length;i++) {
        squareVec.push(vec[i]*vec[i]);
    }
    return squareVec;
};

m.squareMat = function(mat) {
    var squareMat = [];
    var i;
    for(i=0;i<mat.length;i++) {
        squareMat.push(m.squareVec(mat[i]));
    }
    return squareMat;
};

m.minVec = function(vec) {
    var min = vec[0];
    var i = vec.length;
    while (i--) {
        if (vec[i] < min)
            min = vec[i];
    }
    return min;
};

m.maxVec = function(vec) {
    var max = vec[0];
    var i = vec.length;
    while (i--) {
        if (vec[i] > max)
            max = vec[i];
    }
    return max;
}

m.minMat = function(mat) {
    var min = mat[0][0];
    var i = mat.length;
    while (i--) {
        for(var j=0;j<mat[0].length;j++) {
            if(mat[i][j] < min)
                min = mat[i][j];
        }
    }
    return min;
};

m.maxMat = function(mat) {
    var max = mat[0][0];
    var i = mat.length;
    while (i--) {
        for(var j=0;j<mat[0].length;j++) {
            if(mat[i][j] < max)
                max = mat[i][j];
        }
    }
    return max;
};

m.zeroVec = function(n) {
    var vec = [];
    while(vec.length < n)
        vec.push(0);
    return vec;
};

m.zeroMat = function(row,col) {
    var mat = [];
    while(mat.length < row)
        mat.push(m.zeroVec(col));
    return mat;
};

m.oneVec = function(n) {
    var vec = [];
    while(vec.length < n)
        vec.push(1);
    return vec;
};

m.oneMat = function(row,col) {
    var mat = [];
    while(mat.length < row)
        mat.push(m.oneVec(col));
    return mat;
};

m.randVec = function(n,lower,upper) {
    lower = (typeof lower !== 'undefined') ? lower : 0;
    upper = (typeof upper !== 'undefined') ? upper : 1;
    var vec = [];
    while(vec.length < n)
        vec.push(lower + (upper-lower) * Math.random());
    return vec;
};

m.randMat = function(row,col,lower,upper) {
    lower = (typeof lower !== 'undefined') ? lower : 0;
    upper = (typeof upper !== 'undefined') ? upper : 1;
    var mat = [];
    while(mat.length < row)
        mat.push(m.randVec(col,lower,upper));
    return mat;
};

m.randnVec = function(n,mean,sigma) {
    var vec = [];
    while(vec.length < n)
        vec.push(mean+sigma* m.randn());
    return vec;
};

m.randnMat = function(row,col,mean,sigma) {
    var mat = [];
    while(mat.length < row)
        mat.push(m.randnVec(col,mean,sigma));
    return mat;
};

m.identity = function (n) {
    var result = new Array(n);

    for (var i = 0; i < n ; i++) {
        result[i] = new Array(n);
        for (var j = 0; j < n; j++) {
            result[i][j] = (i === j) ? 1 : 0;
        }
    }

    return result;
};

m.sigmoid = function(x) {
    var sigmoid = (1. / (1 + Math.exp(-x)))
    if(sigmoid ==1) {
     //   console.warn("Something Wrong!! Sigmoid Function returns 1. Probably javascript float precision problem?\nSlightly Controlled value to 1 - 1e-14")
        sigmoid = 0.99999999999999; // Javascript Float Precision Problem.. This is a limit of javascript.
    } else if(sigmoid ==0) {
      //  console.warn("Something Wrong!! Sigmoid Function returns 0. Probably javascript float precision problem?\nSlightly Controlled value to 1e-14")
        sigmoid = 1e-14;
    }
    return sigmoid; // sigmoid cannot be 0 or 1;;
};

m.dSigmoid = function(x){
    a = m.sigmoid(x);
    return a * (1. - a);
};

m.probToBinaryMat = function(mat) {
    var row = m.shape(mat)[0];
    var col = m.shape(mat)[1];
    var i,j;
    var result = [];

    for(i=0;i<row;i++) {
        var rowVec = [];
        for(j=0;j<col;j++) {
            if(Math.random() < mat[i][j])
                rowVec.push(1);
            else
                rowVec.push(0);
        }
        result.push(rowVec);
    }
    return result;
};

m.activateVec = function(vec,activation) {
    var i, result = [];
    for(i=0;i<vec.length;i++)
        result.push(activation(vec[i]));
    return result;
};

m.activateMat = function(mat,activation) {
    var row = m.shape(mat)[0];
    var col = m.shape(mat)[1];
    var i, j,result = [];
    for(i=0;i<row;i++) {
        var rowVec = [];
        for(j=0;j<col;j++)
            rowVec.push(activation(mat[i][j]));
        result.push(rowVec);
    }
    return result;
};

m.activateTwoVec = function(vec1, vec2,activation) {
    if (vec1.length === vec2.length) {
        var result = new Array(vec1.length);
        for (var i = 0; i < result.length; i++) {
            result[i] = activation(vec1[i],vec2[i]);
        }
        return result;
    } else {
        throw new Error("Matrix shape error : not same");
    }
};

m.activateTwoMat = function(mat1, mat2,activation) {
    if (mat1.length === mat2.length && mat1[0].length === mat2[0].length) {
        var result = new Array(mat1.length);

        for (var x = 0; x < mat1.length; x++) {
            result[x] = new Array(mat1[0].length);
        }

        for (var i = 0; i < result.length; i++) {
            for (var j = 0; j < result[i].length; j++) {
                result[i][j] = activation(mat1[i][j],mat2[i][j]);
            }
        }
        return result;
    } else {
        throw new Error("Matrix shape error : not same");
    }
};

m.fillVec = function(n,value) {
    var vec = [];
    while(vec.length < n)
        vec.push(value);
    return vec;
};

m.fillMat = function(row,col,value) {
    var mat = [];
    while(mat.length < row) {
        var rowVec = [];
        while(rowVec.length < col)
            rowVec.push(value);
        mat.push(rowVec);
    }
    return mat;
};

m.softmaxVec = function(vec) {
    var max = m.maxVec(vec);
    var preSoftmaxVec = m.activateVec(vec,function(x) {return Math.exp(x - max);})
    return m.activateVec(preSoftmaxVec,function(x) {return x/ m.sumVec(preSoftmaxVec)})
};

m.softmaxMat = function(mat) {
    var result=[], i;
    for(i=0 ; i<mat.length ; i++)
        result.push(m.softmaxVec(mat[i]));
    return result;
};



// For CRBM
/*
m.phi = function(mat,vec,low,high) {
    var i;
    var result = [];
    for(i=0;i<mat.length;i++) {
        result.push(m.activateTwoVec(mat[i],vec,function(x,y){return low+(high-low)* m.sigmoid(x*y);}))
    }
    return result;
}
*/
},{}],11:[function(require,module,exports){
/**
 * Created by joonkukang on 2014. 1. 12..
 */
utils = module.exports;

utils.math = require('./math');
/*

utils.log = function(message,logLevel,type) {
    // log level. 0 : nothing, 1 : info, 2 : warning
    if(logLevel == 1 && type === 'info') {
        console.log(message);
    } else if(logLevel == 2) {
        if(type === 'info')
            console.log(message);
        else if (type === 'warning')
            console.warn(message);
    }

}*/
},{"./math":10}]},{},[1]);
