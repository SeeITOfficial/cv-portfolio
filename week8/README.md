## DAY 1 --> FOURIER TRANSFORM
Images are represented in 2 ways;
1. `Spatial domain` - What we normally work with, a grid of pixel values telling us where things are
2. `Frequency domain` - represents how fast or how rapidly pixel intensitiess change across the image. not where things are but how fast they change 
For images, we will use the `2D Discrete Fourier Transform(DFT)` 

- Consider a black to white gradual fade, thsta a `low frequency signal`; intensity changes slowly
- Consider a chessboard; black and white alternating exery pixel; thats a high intensity change, thats a `high frequency signal`

- Real images contain both
`Low frequencies` = smooth regions, background, gradual color transitions even uniform surfaces or colors
`High frequencies` = edges, fine texture, noise, sharp boundaries

-`Why the center is always the brightest point`
The center of the spectrum (after fftshift) is the `DC component` - it represents the average intensity of the entire image. Since most images have more gradual transitions than sharp edges, low frequencies always dominate. The center is always bright.

## Day 2 --> Low-Pass and High-Pass Filtering in Frequency Domain
In the previous day we converted images into their frequency domain and saw the frequency spectrum and the impact of color intensity, edges and surgaces on the spectrum.
Today we will manipulate the spectrum and reverse all the way back to the image to see the effect.
the workflow ->` image → FFT → shift → apply mask → inverse shift → inverse FFT → result`

- Two types of masks
1. `Low-pass filter` -> keeps the center(low frequency) and blocks out outer region (high freq) Result is a `blurred image` with `edges removed`. Same effect as `Gaussian BLur` but done with frequencies.

2. `High-pass filter` -> block center (low freq), keep outer region (high freq)
Result: `only edges remain`. `Background gone`. Same as what `Sobel/Canny` extract but done with frequencies.

The mask is simply a binary or smooth circular region --> `1` keep the frequencies, `0` block the frequencies.

`After masking you need to get back to spatial domain and the order matters.`
- undo the shift
- inverse fft
- find magnitude(discards tiny imaginary artifacts)

`Hard vs soft masks`
A `hard circular mask` (sharp edge at the cutoff radius) introduces `ringing artifacts` in the output, faint concentric rings around edges. This is called the `Gibbs phenomenon`.
A `soft mask` (Gaussian falloff instead of hard edge) eliminates ringing because the transition is gradual. 