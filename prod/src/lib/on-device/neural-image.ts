import { geometry, normalizedLetterbox, regionalPoolWeights, INPUT_SIZE } from "./neural-contract";

export function prepareImage(cv: typeof import("@techstark/opencv-js"), imageData: ImageData, roi: readonly number[] = [0, 0, 1, 1], ptsOffset = 0) {
  const shape = geometry(imageData.width, imageData.height, roi);
  const source = cv.matFromImageData(imageData);
  const crop = source.roi(new cv.Rect(shape.x, shape.y, shape.cropWidth, shape.cropHeight));
  const rgb = new cv.Mat(), resized = new cv.Mat(), gray = new cv.Mat();
  const floatGray = new cv.Mat(), laplacian = new cv.Mat();
  try {
    cv.cvtColor(crop, rgb, cv.COLOR_RGBA2RGB);
    cv.resize(rgb, resized, new cv.Size(shape.resizedWidth, shape.resizedHeight), 0, 0, cv.INTER_LINEAR);
    const image = normalizedLetterbox(resized.data, shape);
    cv.cvtColor(resized, gray, cv.COLOR_RGB2GRAY);
    // Convert with the same float32 division as NumPy, not convertTo's double scale.
    floatGray.create(gray.rows, gray.cols, cv.CV_32FC1);
    const grayBytes = gray.data, grayFloats = floatGray.data32F;
    let sum = 0, clipped = 0;
    for (let i = 0; i < grayBytes.length; i++) {
      grayFloats[i] = grayBytes[i] / 255;
      sum += grayFloats[i];
      if (grayBytes[i] <= 2 || grayBytes[i] >= 253) clipped++;
    }
    const average = sum / grayBytes.length;
    let variance = 0;
    for (const value of grayFloats) variance += (value - average) ** 2;
    cv.Laplacian(floatGray, laplacian, cv.CV_32F, 1, 1, 0, cv.BORDER_DEFAULT);
    let lapSum = 0;
    for (const value of laplacian.data32F) lapSum += value;
    const lapMean = lapSum / laplacian.data32F.length;
    let lapVariance = 0;
    for (const value of laplacian.data32F) lapVariance += (value - lapMean) ** 2;
    const quality = new Float32Array([shape.resizedWidth * shape.resizedHeight / INPUT_SIZE ** 2,
      average, Math.sqrt(variance / grayBytes.length), lapVariance / laplacian.data32F.length,
      clipped / grayBytes.length, ptsOffset]);
    return { image, quality, poolWeights: regionalPoolWeights(shape.box), shape };
  } finally {
    laplacian.delete(); floatGray.delete(); gray.delete(); resized.delete(); rgb.delete(); crop.delete(); source.delete();
  }
}
