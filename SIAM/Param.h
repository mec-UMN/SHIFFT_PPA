/*******************************************************************************
* Copyright (c) 2017-2020
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Yu Cao
* All rights reserved.
*
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory).
* Copyright of the model is maintained by the developers, and the model is distributed under
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
*
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer.
*
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* Developer list:
*		Gokul Krishnan Email: gkrish19@asu.edu

* Credits: Prof.Shimeng Yu and his research group for NeuroSim
********************************************************************************/

#ifndef PARAM_H_
#define PARAM_H_

class Param {
public:
	Param();

	int operationmode, memcelltype, accesstype, transistortype, deviceroadmap, max_little_count, max_big_count,max_mid_count;

	double heightInFeatureSizeSRAM, widthInFeatureSizeSRAM, widthSRAMCellNMOS, widthSRAMCellPMOS, widthAccessCMOS, minSenseVoltage;

	double heightInFeatureSize1T1R, widthInFeatureSize1T1R, heightInFeatureSizeCrossbar, widthInFeatureSizeCrossbar;

	int relaxArrayCellHeight, relaxArrayCellWidth, algoWeightMax, algoWeightMin, pool_stride;

	bool globalBufferType, tileBufferType, peBufferType, chipActivation, reLu, novelMapping, mode, chiplet_structure, fixed_cnt_chiplet, currentMode, inputdacmode;

	double clkFreq, featuresize, readNoise, resistanceOn, resistanceOff, maxConductance, minConductance;
	int temp, technode, wireWidth, multipleCells, layer_parallel;
	double maxNumLevelLTP, maxNumLevelLTD, readVoltage, readPulseWidth, writeVoltage;
	double accessVoltage, resistanceAccess;
	double nonlinearIV, nonlinearity;
	double writePulseWidth, numWritePulse;
	double globalBusDelayTolerance, localBusDelayTolerance;
	double treeFoldedRatio, maxGlobalBusWidth;
	double sparsity;

	int tileBufferCoreSizeRow, tileBufferCoreSizeCol, globalBufferCoreSizeRow, globalBufferCoreSizeCol;

	int neuro, multifunctional, parallelWrite, parallelRead;
	int numlut, numColMuxed, numWriteColMuxed, levelOutput, avgWeightBit, numBitInput, Chiplet_GAccumulator;
	int numRowSubArray, numColSubArray, row_multiplier, col_multiplier, to_subtract, numRowSubArray_big, numColSubArray_big, numRowSubArray_little, numColSubArray_little,numRowSubArray_mid, numColSubArray_mid,numRowSubArray_type1,numRowSubArray_type2,numRowSubArray_type3
	,numRowSubArray_type4,numRowSubArray_type5,numRowSubArray_type6,numColSubArray_type1,numColSubArray_type2,numColSubArray_type3,numColSubArray_type4,numColSubArray_type5,numColSubArray_type6;
	int cellBit, synapseBit, size_chiplet, cnt_chiplet, size_chiplet_big, size_chiplet_little,size_chiplet_mid,size_chiplet_1,size_chiplet_2,size_chiplet_3,size_chiplet_4,size_chiplet_5,size_chiplet_6;

	int XNORparallelMode, XNORsequentialMode, BNNparallelMode, BNNsequentialMode, conventionalParallel, conventionalSequential;
	int numRowPerSynapse, numColPerSynapse;
	double AR, Rho, wireLengthRow, wireLengthCol, unitLengthWireResistance, wireResistanceRow, wireResistanceCol;
};

#endif
