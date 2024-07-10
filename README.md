# neuro-radar-snn
SNN training and testing using the spike data from multi-channel NeuroRadar sensor. 

The SNN construction and optimization is based on the [nengo_dl](https://www.nengo.ai/nengo-dl/) framework. 

# SNN Structure
![image](https://github.com/kaizheng28/neuro-radar-snn/assets/144567523/5764d51e-0c3e-4e9b-8c80-b507c6a68ea3)

In our work, the spike buffering units are implemented in software. The dataset has already been processed to produce spikes after the buffering unit (unit delay = 4ms).
In the current implementation, the whole spike sequence (1.5s for gesture recognition, and 2s for localization) needs to be buffered before they enter the next layer. 
A more detailed description can be found in Sec.5.2 of the NeuroRadar paper. 

# Future works
Nengo-dl only supports the conversion method for training SNNs. This method involves training a traditional deep neural network (DNN) with the same structure and then converting it into an SNN. 
The DNN-SNN conversion method has limitations. The conversion method fails to incorporate the inherent temporal dynamics of the spiking neuron models. 

As a result, firstly, it can lead to a high spike rate since it is essentially using spike density to represent a continuous value in the activation layer. A high spike rate usually means higher power consumption. 
Secondly, it is also not best suited for data that are directly produced by neuromorphic sensors which inherently encode information in the temporal domain. 
Therefore, in our design, spike buffering units must be added to flatten the temporal dimension and into the spatial dimension so that the feature can be captured by the convolutional layers. 
Buffering leads to large latency and extra hardware complexity; this is also not ideal. 

For future works, other optimization methods such as Back-Propagation Through Time (BPTT) which considers the temporal features can be explored to process the NeuroRadar data. 
Other SNN structures such as [Liquid State Machine (LSM)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.819063/full) are also worth investigating. 
Spike-timing-dependent plasticity (STDP) is also an interesting online training method that is worth looking into.

Recommended framework: [SNNTorch](https://snntorch.readthedocs.io/en/latest/readme.html).

# More resources
[Spike sampler FPGA program and raw dataset](https://github.com/kaizheng28/spike-sampler)

[Hardware design files](https://github.com/kaizheng28/neuro-radar-pcb)

# References
Kai Zheng, Kun Qian, Timothy Woodford, and Xinyu Zhang. 2024. NeuroRadar: A Neuromorphic Radar Sensor for Low-Power IoT Systems. In Proceedings of the 21st ACM Conference on Embedded Networked Sensor Systems (SenSys '23). Association for Computing Machinery, New York, NY, USA, 223–236. https://doi.org/10.1145/3625687.3625788
