This folder contains the optimization passes onnx2c runs.

Due to lack of up-front design, and surplus development time later,
these passes are not stand-alone, but rather a part of the (ever
increasingly pointless) Graph object.

Setting up a mechanism of stand-alone optimization passes that can
mutate the Graph via a public API sounds like a cleanup task that
would make the code look much cleaner.

But add a few optimization passes before starting to design such an API.

