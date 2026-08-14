package com.volleycut.nativeanalysis;

final class Radix2Fft {
    private final int size;
    private final int levels;
    private final double[] cos;
    private final double[] sin;

    Radix2Fft(int size) {
        if (Integer.bitCount(size) != 1) throw new IllegalArgumentException("FFT size must be a power of two");
        this.size = size;
        this.levels = 31 - Integer.numberOfLeadingZeros(size);
        this.cos = new double[size / 2];
        this.sin = new double[size / 2];
        for (int i = 0; i < size / 2; i++) {
            cos[i] = Math.cos(2 * Math.PI * i / size);
            sin[i] = Math.sin(2 * Math.PI * i / size);
        }
    }

    float[] magnitudes(float[] input) {
        if (input.length != size) throw new IllegalArgumentException("FFT input size mismatch");
        double[] real = new double[size];
        double[] imaginary = new double[size];
        for (int i = 0; i < size; i++) {
            int reversed = Integer.reverse(i) >>> (32 - levels);
            real[reversed] = input[i];
        }
        for (int block = 2; block <= size; block <<= 1) {
            int half = block >>> 1;
            int tableStep = size / block;
            for (int start = 0; start < size; start += block) {
                for (int j = 0; j < half; j++) {
                    int table = j * tableStep;
                    int even = start + j;
                    int odd = even + half;
                    double oddReal = real[odd] * cos[table] + imaginary[odd] * sin[table];
                    double oddImaginary = -real[odd] * sin[table] + imaginary[odd] * cos[table];
                    real[odd] = real[even] - oddReal;
                    imaginary[odd] = imaginary[even] - oddImaginary;
                    real[even] += oddReal;
                    imaginary[even] += oddImaginary;
                }
            }
        }
        float[] magnitudes = new float[size / 2 + 1];
        for (int i = 0; i < magnitudes.length; i++) {
            magnitudes[i] = (float) Math.hypot(real[i], imaginary[i]);
        }
        return magnitudes;
    }
}
