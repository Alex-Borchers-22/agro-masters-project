/**
 * Handles reformating image by changing rbg channels
 * @author Alex Borchers
 * @param {HTMLElement} ctx (canvas object)
 * @param {HTMLElement} imageData 
 * @param {string} constant (rbg/bgr/brg/gbr/grb/rbg)
 * @returns {HTMLElement} newly formatted canvas object
 * 
*/
function reformat_rgb(ctx, imageData, rgb_order) {

    // Validate rgb_order
    if (!validate_rgb(rgb_order)){
        alert("[Error] The RGB parameter does not meet the given criteria.");
        return;
    }

    // Get the image data from the canvas
    var data = imageData.data;

    // Get index of rgb channels
    const rIndex = rgb_order.indexOf('r');
    const gIndex = rgb_order.indexOf('g');
    const bIndex = rgb_order.indexOf('b');

    // Modify the RGB values to switch red and blue channels
    for (let i = 0; i < data.length; i += 4) {

        // Get red, green & blue channels
        var red = data[i + rIndex];
        var green = data[i + gIndex];
        var blue = data[i + bIndex];

        // Update to new channels
        data[i] = red;
        data[i + 1] = green;
        data[i + 2] = blue;
    }

    // Put the modified image data back onto the canvas
    ctx.putImageData(imageData, 0, 0);

    // Return modified object
    return ctx;
}

/**
 * Handles validating rgb_order syntax
 * @param {string} rgb_order (order of channels)
 * @returns {boolean} false=error
*/
function validate_rgb(rgb_order){
    // Set value keys & check for index
    var check_keys = ["rgb", "bgr", "brg", "gbr", "grb", "rbg"];
    return check_keys.includes(rgb_order);
}

/**
 * Handles reformating image by multiplying input channel by constant
 * @author Alex Borchers
 * @param {HTMLElement} ctx (canvas object)
 * @param {HTMLElement} imageData (canvas object)
 * @param {string} constant (rbg/bgr/brg/gbr/grb/rbg)
 * @returns {HTMLElement} newly formatted canvas object
 * 
*/
function reformat_by_constant(ctx, imageData, constant) {

    // Get the image data from the canvas
    var data = imageData.data;

    // Modify the channels multiplying by a constant
    for (let i = 0; i < data.length; i++) {
        data[i] = parseInt(data[i]) * parseFloat(constant);
    }

    // Put the modified image data back onto the canvas
    ctx.putImageData(imageData, 0, 0);

    // Return modified object
    return ctx;
}

/**
 * Handles reformating image by transforming canvas object
 * @author Alex Borchers
 * @param {HTMLElement} ctx (canvas object)
 * @param {HTMLElement} imageData (canvas object)
 * @param {string} transform (rotate90/rotate180/rotate270/mirror)
 * @returns {HTMLElement} newly formatted canvas object
 * 
*/
function reformat_by_transform(ctx, imageData, transform, update_img) {

    // Get the image data from the canvas
    var data = imageData.data;

    // Create temporary canvas for image mirroring
    var temp_canvas = document.createElement('canvas');
    temp_canvas.width = c_width;
    temp_canvas.height = c_height;
    var temp_ctx = temp_canvas.getContext('2d');

    // Apply the transform to the canvas
    switch(transform) {
        case 'rotate90':
            // Rotate image 90 degrees
            temp_ctx.translate(c_height, 0);
            temp_ctx.rotate(Math.PI / 2);
            temp_ctx.drawImage(ctx.canvas, 0, 0);

            // Update image data with rotated image data
            imageData = temp_ctx.getImageData(0, 0, c_height, c_width);
            data = imageData.data;

            // Adjust object position so overlay is hidden
            update_img.style.objectPosition = "-100% 0";

            break;
        case 'rotate180':
            // Rotate image 180 degrees
            temp_ctx.translate(c_width, c_height);
            temp_ctx.rotate(Math.PI);
            temp_ctx.drawImage(ctx.canvas, 0, 0);

            // Update image data with rotated image data
            imageData = temp_ctx.getImageData(0, 0, c_width, c_height);
            data = imageData.data;

            // Adjust object position so overlay is hidden
            update_img.style.objectPosition = "";

            break;
        case 'rotate270':
            // Rotate image 270 degrees
            temp_ctx.translate(0, c_width);
            temp_ctx.rotate(-Math.PI / 2);
            temp_ctx.drawImage(ctx.canvas, 0, 0);

            // Update image data with rotated image data
            imageData = temp_ctx.getImageData(0, 0, c_height, c_width);
            data = imageData.data;

            // Adjust object position so overlay is hidden
            update_img.style.objectPosition = "-100% 0";

            break;
        case 'mirror':
            // Mirror image
            temp_ctx.translate(c_width, 0);
            temp_ctx.scale(-1, 1);
            temp_ctx.drawImage(ctx.canvas, 0, 0);

            // Update image data with mirrored image data
            imageData = temp_ctx.getImageData(0, 0, c_width, c_height);
            data = imageData.data;

            // Adjust object position so overlay is hidden
            update_img.style.objectPosition = "";

            break;
        default:
            update_img.style.objectPosition = "";
            break;
    }

    // Put the modified image data back onto the canvas
    ctx.putImageData(imageData, 0, 0);

    return ctx;

}

/**
 * Handles reformating image by normalizing data
 * @author Alex Borchers
 * @param {HTMLElement} ctx (canvas object)
 * @param {HTMLElement} imageData (canvas object)
 * @returns {HTMLElement} newly formatted canvas object
 * 
*/
function reformat_by_normalizing_data(ctx, imageData) {

    // Get the image data from the canvas
    var data = imageData.data;

    // Get standard deviation of data to use as a devisor
    var std_dev = getStandardDeviation(data);
    //alert(std_dev);		//15.366...

    for (var i = 0; i < data.length; i++) {
        data[i] = parseFloat(data[i]) / std_dev;
    }

    // Put the modified image data back onto the canvas
    ctx.putImageData(imageData, 0, 0);

    // Return modified object
    return ctx;
}

/**
 * Handles reformating image by inverting data
 * @author Alex Borchers
 * @param {HTMLElement} ctx (canvas object)
 * @param {HTMLElement} imageData (canvas object)
 * @returns {HTMLElement} newly formatted canvas object
 * 
*/
function reformat_by_inverting_data(ctx, imageData) {

    // Get the image data from the canvas
    var data = imageData.data;

    // Modify the input channel by changing to new assignments
    for (var i = 0; i < data.length; i+=4) {
        data[i] = 255 - data[i];       // red
        data[i + 1] = 255 - data[i + 1]; // green
        data[i + 2] = 255 - data[i + 2]; // blue
    }

    // Put the modified image data back onto the canvas
    ctx.putImageData(imageData, 0, 0);

    // Return modified object
    return ctx;

}

/**
 * Handles reformating image by permutating image data
 * @author Alex Borchers
 * @param {HTMLElement} ctx (canvas object)
 * @param {HTMLElement} imageData (canvas object)
 * @returns {HTMLElement} newly formatted canvas object
 * 
*/
function reformat_by_permutation(ctx, imageData) {

    // Get the image data from the canvas
    var data = imageData.data;

    // Get # of potential input channels (divide total by 3)
    var pot_channels = data.length / 3;

    // Create original index of array
    var indexes = [];
    for (var i = 0; i < pot_channels; i++){
        indexes.push(i);
    }

    console.log(pot_channels)

    // Randomly shuffle array
    indexes = shuffle_array(indexes)

    // Modify the input channel by changing to new assignments
    for (var i = 0; i < indexes.length; i++) {
        data[3*indexes[i]] = data[3*i];
        data[3*indexes[i] + 1] = data[3*i + 1];
        data[3*indexes[i] + 2] = data[3*i + 2];
    }

    // Put the modified image data back onto the canvas
    ctx.putImageData(imageData, 0, 0);

    // Return modified object
    return ctx;
}

// Handles randomly shuffling an array 
// https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-array
function shuffle_array(array) {
    let currentIndex = array.length,  randomIndex;

    // While there remain elements to shuffle.
    while (currentIndex != 0) {

        // Pick a remaining element.
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;

        // And swap it with the current element.
        [array[currentIndex], array[randomIndex]] = [
        array[randomIndex], array[currentIndex]];
    }

    return array;
}

// Source https://stackoverflow.com/questions/7343890/standard-deviation-javascript 
function getStandardDeviation (array) {
    const n = array.length
    const mean = array.reduce((a, b) => a + b) / n
    return Math.sqrt(array.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / n)
}