
This package is intended to be used as a tool combined with the API for plecomusic.com.

```javascript
// Create a CTRNN.
var ctrnn = new CTRNN();

// Pass in your CTRNN configuration.
ctrnn.setConfiguration(config);

// Initialise network with new  .
ctrnn.initialise(deltaTime);

// Feed input into network. Each element of the array is fed into all CTRNN Input Neurons.
// Length of array must match the number of CTRNN inputs.
ctrnn.feedInputs(inputArray);

// Update network state.
ctrnn.update();

// Get network output as an array.
var outputs = ctrnn.getOutputs();

// Resets CTRNN.
ctrnn.reset();
```

MIT License

Copyright (c) 2020 Steffan Ianigro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
