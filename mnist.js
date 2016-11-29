

// Original files were compiled into batches for easy browser access by @karpathy.
// See : http://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist/
// Load the dataset with JS in background
var data_img_elts = [null, null, null, null, null, null];
var img_data = [null, null, null, null, null];
var loaded = [null, null, null, null, null];


var MNIST = function(folder_path) {
  for (var i=0; i<20; i++) {
    this.data_img_elts.push(null);
    this.img_data.push(null);
    this.loaded.push(null);
  };
  this.folder_path = folder_path;
};

MNIST.prototype.folder_path = null; //"mnist";
MNIST.prototype.data_img_elts = [];
MNIST.prototype.img_data = [];
MNIST.prototype.loaded = [];


MNIST.prototype.load_data_batch = function(batch_num, verbose) {
  this.data_img_elts[batch_num] = new Image();
  var data_img_elt = data_img_elts[batch_num];
  data_img_elt.onload = function() { 
    //var data_canvas = document.getElementById('datacanvas');
    var data_canvas = document.createElement('canvas');
    data_canvas.width = data_img_elt.width;
    data_canvas.height = data_img_elt.height;
    var data_ctx = data_canvas.getContext("2d");
    data_ctx.drawImage(data_img_elt, 0, 0); // copy it over... bit wasteful :(
    this.img_data[batch_num] = data_ctx.getImageData(0, 0, data_canvas.width, data_canvas.height);
    // WARNING : the data array length is equal to 4*width*height
    this.loaded[batch_num] = true;
    if (verbose) {
      console.log('finished loading data batch ' + batch_num);
      console.log('WARNING : should remove the new canvas?');
    }
  };
  data_img_elt.src = this.folder_path + "/mnist_batch_" + batch_num + ".png";
};


MNIST.prototype.mnistitem = function(batch_num, row) {
  // var batch_num = 1;
  var p = this.img_data[batch_num].data;
  //var row = 15; // any number from 0 to 2999, each row correspond to a digit image
  var W = 28*28;
  var vec = [];
  for(var i=0; i<W; i++) {
    var ix = ((W * row) + i) * 4; // WARNING : the data array length is equal to 4*width*height
    vec.push( p[ix]/255.0 ); // scale the data
  }
  return vec;
};

