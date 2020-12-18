---
title: Online Mel Spectrum Analyzer
published: true
---
<meta name="Online web based Mel-spectrum, power spectrum, FFT analyzer for speech and music processing.">

# [](#Mel-Spectrogram)Mel Spectrogram


![png](https://tabahi.github.io/assets/posts/Online-Mel-Spectrum-Analyzer-1.png)

Check it out here: [Online Mel Spectrum Analyzer](https://tabahi.github.io/Mel-Spectrum-Analyzer/)

It is fairly convenient to extract [Mel Spectrogram in Matlab](https://www.mathworks.com/help/audio/ref/melspectrogram.html) or [Python](https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html) by a single command. However, online web applications are limited by the JavaScript engine capabilities of web browsers. Few years ago, Firefox and Chrome introduced the [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API) which handles scripted the audio operations within the browser context. Most of the browsers currently support basic Web Audio API features. But the advanced level processing features are still limited. For example, and FFT node outputs the FFT but there are not many parameters to customize the FFT window and sampling rate.


![png](https://www.mathworks.com/help/audio/ref/melspectrogram_2.png)

# [](#Audio-Worklet)Audio Worklet

With the recent upgrades, Firefox and Chrome has introduced an [Audio Worklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorkletNode) that runs a separate thread for handling a customizable audio processing node. I have used this feature to program a customized FFT and then use the FFT to create a Mel-Spectrogram.

One of the big difference between the built-in FFT node and this customized AudioWorklet based FFT node is that it gives us a lot more windowing options:


![png](https://tabahi.github.io/assets/posts/Online-Mel-Spectrum-Analyzer-2.png)

But there are few drawbacks to it. Since this customized node is scripted in JavaScript (A high level language) and the built-in FFT node is, well, built in most probably in C++, a low level language which is why works much faster than the customized node. But that can be solved too by using [WebAssembly](https://developer.mozilla.org/en-US/docs/WebAssembly).

Most of the smartphone Apps are also opting to use the JS engines because of the portability of source code. The Native web engines of smartphones already support most of the Web API. If someone develops a WASM based plugin for React or Angular or Vue, then it will help create many more Audio Processing Apps in JS.

