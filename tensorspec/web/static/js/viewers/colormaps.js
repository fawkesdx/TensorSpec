/* Colormap lookup tables.
 *
 * Colour is a display choice, so it lives in the browser rather than in the
 * Python core. Each map is defined by evenly spaced control points that are
 * interpolated into a 256-entry table once, at module load.
 */

const CONTROL_POINTS = {
    magma: [
        [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122],
        [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191],
    ],
    viridis: [
        [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
        [31, 158, 137], [53, 183, 121], [109, 205, 89], [253, 231, 37],
    ],
    inferno: [
        [0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110], [147, 38, 103],
        [188, 55, 84], [221, 81, 58], [243, 120, 25], [252, 255, 164],
    ],
    gray: [[0, 0, 0], [255, 255, 255]],
};

function buildTable(points) {
    const table = new Uint8ClampedArray(256 * 3);
    const spans = points.length - 1;

    for (let i = 0; i < 256; i++) {
        const position = (i / 255) * spans;
        const lower = Math.min(Math.floor(position), spans - 1);
        const blend = position - lower;
        for (let channel = 0; channel < 3; channel++) {
            const a = points[lower][channel];
            const b = points[lower + 1][channel];
            table[i * 3 + channel] = a + (b - a) * blend;
        }
    }
    return table;
}

export const COLORMAPS = Object.fromEntries(
    Object.entries(CONTROL_POINTS).map(([name, points]) => [name, buildTable(points)])
);

export const COLORMAP_NAMES = Object.keys(COLORMAPS);

/* A short horizontal strip for colourbar swatches. */
export function colormapStripe(name, width = 120, height = 12) {
    const table = COLORMAPS[name] || COLORMAPS.magma;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const image = canvas.getContext("2d").createImageData(width, height);

    for (let x = 0; x < width; x++) {
        const index = Math.round((x / (width - 1)) * 255) * 3;
        for (let y = 0; y < height; y++) {
            const offset = (y * width + x) * 4;
            image.data[offset] = table[index];
            image.data[offset + 1] = table[index + 1];
            image.data[offset + 2] = table[index + 2];
            image.data[offset + 3] = 255;
        }
    }
    canvas.getContext("2d").putImageData(image, 0, 0);
    return canvas.toDataURL();
}
