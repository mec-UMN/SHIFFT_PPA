# SHIFFT_PPA

SHIFFT is a a scalable hybrid in-memory computing FFT accelerator (SHIFFT), a hybrid architecture that combines RRAM-based in-memory computing with CMOS adders. This architecture was developed to overcome the latency limitations seen in traditional CMOS accelerators. The tool is developed to estimate the EDP and FFT throughput for SHIFFT architecture. The architecture is as follows: the chip architecture comprises an array of IMC tiles, adder trees, and buffers connected using HTree. Each IMC tile is composed of a 2x2 array of processing elements (PE) and an adder tree stage that combines the PE outputs in the same column within the PE array. Each PE, in turn, contains a 2x2 array of IMC crossbar arrays (Xbar) and an adder tree stage that adds the crossbar outputs in the same column within the array of crossbars.

![SHIFFT Architecture](https://github.com/mec-UMN/SHIFFT_PPA/blob/main/SHIFFT%20architecture.jpg)

The tool was developed by modifying the SIAM benchmarking tool, that was initially developed for machine learning application, to suit the needs of DFT and proposed SHIFFT architecture.
## Usage

## References
```
Pragnya Sudershan Nalla, Zhenyu Wang, Sapan Agarwal, T. Patrick Xiao, Christopher H. Bennett, Matthew J. Marinella, Jae-sun Seo, and Yu Cao, SHIFFT: A Scalable Hybrid In-Memory Computing FFT Accelerator, ISVLSI 2024

@article{Krishnan_2021,
   title={SIAM: Chiplet-based Scalable In-Memory Acceleration with Mesh for Deep Neural Networks},
   volume={20},
   ISSN={1558-3465},
   url={http://dx.doi.org/10.1145/3476999},
   DOI={10.1145/3476999},
   number={5s},
   journal={ACM Transactions on Embedded Computing Systems},
   publisher={Association for Computing Machinery (ACM)},
   author={Krishnan, Gokul and Mandal, Sumit K. and Pannala, Manvitha and Chakrabarti, Chaitali and Seo, Jae-Sun and Ogras, Umit Y. and Cao, Yu},
   year={2021},
   month=sep, pages={1–24} }
```

## Developers
Main devs:
* Pragnya Nalla 

Advisors
* Matthew J. Marinella
* Jae-sun Seo
* Yu Cao
