
var meanshift = function (points, kvariance, nbiterations) {

	var nb = numeric.dim(points)[0];
	console.log(nb);

	var kernel = function(X, Y) {
  		var dA = numeric.dotVV(numeric.sub(X,Y), numeric.sub(X,Y));
  		return Math.exp(-0.5*dA/kvariance);
  	}
	
	for (var iter=0; iter<nbiterations; iter++) {
		console.log(iter);

		var mean_shifts = numeric.mul(numeric.clone(points), 0);

		for (var i=0; i<nb; i++) {
	  		var mp = numeric.mul(points[i], 0); // mean shift vector
	  		//console.log(mp);
	  		var sw = 0; // sum of weights
	  		for (var j=0; j<nb; j++) {
	  			var wgt = kernel(points[j], points[i]);
	  			mp = numeric.add(mp, numeric.mul(points[j], wgt));
	  			sw+= wgt;
	  		}
	  		//console.log(sw);
	  		mp = numeric.div(mp, sw);
	  		mean_shifts[i] = numeric.sub(mp, points[i]);
	  	}

	  	for (var i=0; i<nb; i++) {
	  		points[i] = numeric.add(mean_shifts[i], points[i]);
	  	}
	}
	console.log('done');
	return points;
  	
};


// INFO :
// The following "grouping" code has been modified from Python code at :
// https://github.com/mattnedrich/MeanShift_py
// Thanks
var group_points = function (points, GROUP_DISTANCE_TOLERANCE) { // optional GROUP_DISTANCE_TOLERANCE
	if (GROUP_DISTANCE_TOLERANCE == null) {
		GROUP_DISTANCE_TOLERANCE = 0.1;
	}
    group_assignment = [];
    groups = [];
    group_index = 0;
    index = 0;
    points.forEach(function(point) {
        nearest_group_index = _determine_nearest_group(point, groups, GROUP_DISTANCE_TOLERANCE)
        if (nearest_group_index == null) {
            // create new group
            groups.push([point]);
            group_assignment.push(group_index);
            group_index += 1;
        }
        else {
            group_assignment.push(nearest_group_index);
            groups[nearest_group_index].push(point);
        }
        index += 1;
    });
    console.log( Math.max.apply(Math, group_assignment) );
    return group_assignment;
}


var _determine_nearest_group = function (point, groups, GROUP_DISTANCE_TOLERANCE) {
    nearest_group_index = null;
    index = 0;
    groups.forEach(function(group) {
        distance_to_group = _distance_to_group(point, group);
        if (distance_to_group < GROUP_DISTANCE_TOLERANCE) {
            nearest_group_index = index;
        }
        index += 1
    });
    return nearest_group_index;
}

var _euclidean_dist = function (pointA, pointB) {
    if (pointA.length != pointB.length) {
        throw("expected point dimensionality to match");
    }
    total = 0.;
    for (var dimension=0; dimension<pointA.length; dimension++) {
        total += Math.pow(pointA[dimension] - pointB[dimension], 2);
    }
    return Math.sqrt(total)
}

var _distance_to_group = function (point, group) {
    min_distance = Number.MAX_VALUE;
    group.forEach(function(pt) {
        dist = _euclidean_dist(point, pt);
        if (dist < min_distance) {
            min_distance = dist;
        }
    });
    return min_distance;
}

