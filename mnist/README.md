# MNIST data


Original files were compiled into batches for easy browser access by @karpathy.

See : http://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist/

Original source : http://yann.lecun.com/exdb/mnist/

Python script for batch generation with explanation: http://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html


```javascript
// code snippet from @karpathy to turn extract data from batch png files
var load_data_batch = function(batch_num) {
  // Load the dataset with JS in background
  data_img_elts[batch_num] = new Image();
  var data_img_elt = data_img_elts[batch_num];
  data_img_elt.onload = function() { 
    var data_canvas = document.createElement('canvas');
    data_canvas.width = data_img_elt.width;
    data_canvas.height = data_img_elt.height;
    var data_ctx = data_canvas.getContext("2d");
    data_ctx.drawImage(data_img_elt, 0, 0); // copy it over... bit wasteful :(
    img_data[batch_num] = data_ctx.getImageData(0, 0, data_canvas.width, data_canvas.height);
    loaded[batch_num] = true;
    if(batch_num < 20) { loaded_train_batches.push(batch_num); }
    console.log('finished loading data batch ' + batch_num);
  };
  data_img_elt.src = "mnist/mnist_batch_" + batch_num + ".png";
}
```

```javascript
// fetch the appropriate row of the training image and reshape into a Vol
var p = img_data[b].data;
var row = 15 // any number from 0 to 2999, each row correspond to a digit image
var W = 28*28;
for(var i=0;i<W;i++) {
  var ix = ((W * k) + i) * 4;
  p[ix]/255.0;
}
```
