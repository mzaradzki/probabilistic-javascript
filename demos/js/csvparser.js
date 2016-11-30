
function csvParser(allText) {
    var allTextLines = allText.split(/\r\n|\n/);
    var headers = allTextLines[0].split(',');
    for (var j = 0; j < headers.length; j++) {
        headers[j] = eval( headers[j] ); // "eval" to remove double quote
    }
    //console.log(headers);
    var columns = {};
    for (var j = 0; j < headers.length; j++) {
        columns[headers[j]] = [];
    }
    for (var i = 1; i < allTextLines.length; i++) {
        var data = allTextLines[i].split(',');
        if (data.length == headers.length) {
            //var tarr = [];
            for (var j = 0; j < headers.length; j++) {
                //tarr.push(headers[j] + ":" + data[j]);
                columns[headers[j]].push( eval(data[j]) ); // "eval" to remove double quote
            }
            //lines.push(tarr);
        }
    }
    //alert(columns);
    return columns;
}