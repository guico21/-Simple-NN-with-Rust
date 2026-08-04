## Improvements
- [ ] Update functions to be Result<T,E> based. This is to ensure the management of failures
- [ ] Memory optimization. Removing data copying where possible
- [ ] Test for backward layer
- [ ] test for gradient
- [ ] RefCell and Rc for reducing allocation of vectors during backwards propagation?
- [ ] Make the values of images and size parametric. right now is all hard coded.
- [ ] Memory management when loading the images. Instead of loading everything in memory, divide the loading -> training phase with the data we need. This is slower but help in memory management.
- [ ] Add better tests once the hard coded values are made parametric. currently are locked with 10k and 784
- [ ] 

## Documentation
- [ ] Add project structure definiton (once done the core architecture and tested the main functionalities)
- [ ] Explanation of He matrix initialization
- [ ] cross-entropy explanation with softmax. formulas included
- [ ] my matrix definition is row major
- [ ] argmax explanation
