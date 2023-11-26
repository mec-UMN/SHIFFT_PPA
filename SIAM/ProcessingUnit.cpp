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

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include "Bus.h"
#include "SubArray.h"
#include "constant.h"
#include "formula.h"
#include "ProcessingUnit.h"
#include "Param.h"
#include "AdderTree.h"
#include "Bus.h"
#include "DFF.h"

using namespace std;

extern Param *param;

AdderTree *adderTree;
Bus *busInput;
Bus *busOutput;
DFF *bufferInput;
DFF *bufferOutput;

void ProcessingUnitInitialize(SubArray *& subArray, InputParameter& inputParameter, Technology& tech, MemCell& cell, int _numSubArrayRow, int _numSubArrayCol, double desiredIMCsize_x, double desiredIMCsize_y) {

	/*** circuit level parameters ***/
	switch(param->memcelltype) {
		case 3:     cell.memCellType = Type::FeFET; break;
		case 2:	    cell.memCellType = Type::RRAM; break;
		case 1:	    cell.memCellType = Type::SRAM; break;
		case -1:	break;
		default:	exit(-1);
	}
	switch(param->accesstype) {
		case 4:	    cell.accessType = none_access;  break;
		case 3:	    cell.accessType = diode_access; break;
		case 2:	    cell.accessType = BJT_access;   break;
		case 1:	    cell.accessType = CMOS_access;  break;
		case -1:	break;
		default:	exit(-1);
	}

	switch(param->transistortype) {
		case 3:	    inputParameter.transistorType = TFET;          break;
		case 2:	    inputParameter.transistorType = FET_2D;        break;
		case 1:	    inputParameter.transistorType = conventional;  break;
		case -1:	break;
		default:	exit(-1);
	}

	switch(param->deviceroadmap) {
		case 2:	    inputParameter.deviceRoadmap = LSTP;  break;
		case 1:	    inputParameter.deviceRoadmap = HP;    break;
		case -1:	break;
		default:	exit(-1);
	}

	subArray = new SubArray(inputParameter, tech, cell);
	adderTree = new AdderTree(inputParameter, tech, cell);
	busInput = new Bus(inputParameter, tech, cell);
	busOutput = new Bus(inputParameter, tech, cell);
	bufferInput = new DFF(inputParameter, tech, cell);
	bufferOutput = new DFF(inputParameter, tech, cell);

	/* Create SubArray object and link the required global objects (not initialization) */
	inputParameter.temperature = param->temp;   // Temperature (K)
	inputParameter.processNode = param->technode;    // Technology node
	tech.Initialize(inputParameter.processNode, inputParameter.deviceRoadmap, inputParameter.transistorType);

	cell.resistanceOn = param->resistanceOn;	                                // Ron resistance at Vr in the reported measurement data (need to recalculate below if considering the nonlinearity)
	cell.resistanceOff = param->resistanceOff;	                                // Roff resistance at Vr in the reported measurement dat (need to recalculate below if considering the nonlinearity)
	cell.resistanceAvg = (cell.resistanceOn + cell.resistanceOff)/2;            // Average resistance (for energy estimation)
	cell.readVoltage = param->readVoltage;	                                    // On-chip read voltage for memory cell
	cell.maxNumLevelLTP = param->maxNumLevelLTP;	                            // Maximum number of conductance states during LTP or weight increase
	cell.maxNumLevelLTD = param->maxNumLevelLTD;	                            // Maximum number of conductance states during LTD or weight decrease
	double writeVoltageLTP = param->writeVoltage;
	double writeVoltageLTD = param->writeVoltage;
	cell.writeVoltage = sqrt(writeVoltageLTP * writeVoltageLTP + writeVoltageLTD * writeVoltageLTD);    // Use an average value of write voltage for NeuroSim
	cell.readPulseWidth = param->readPulseWidth;
	double writePulseWidthLTP = param->writePulseWidth;
	double writePulseWidthLTD = param->writePulseWidth;
	cell.writePulseWidth = (writePulseWidthLTP + writePulseWidthLTD) / 2;
	cell.accessVoltage = param->accessVoltage;                                       // Gate voltage for the transistor in 1T1R
	cell.resistanceAccess = param->resistanceAccess;
	cell.multipleCells = param->multipleCells;                                       // Value should be N^2 such as 1, 4, 9 ...etc
	cell.featureSize = param->featuresize;
	cell.nonlinearIV = param->nonlinearIV;                                           // This option is to consider I-V nonlinearity in cross-point array or not
	cell.nonlinearity = param->nonlinearity;                                             // This is the nonlinearity for the current ratio at Vw and Vw/2
	if (cell.nonlinearIV) {
		double Vr_exp = 1;	                                            // XXX: Modify this to Vr in the reported measurement data (can be different than cell.readVoltage)
		// Calculation of resistance at on-chip Vr
		cell.resistanceOn = NonlinearResistance(cell.resistanceOn, cell.nonlinearity, cell.writeVoltage, Vr_exp, cell.readVoltage);
		cell.resistanceOff = NonlinearResistance(cell.resistanceOff, cell.nonlinearity, cell.writeVoltage, Vr_exp, cell.readVoltage);
		cell.resistanceAvg = (cell.resistanceOn + cell.resistanceOff)/2;      // Average resistance (for energy estimation)
	}

	if (cell.memCellType == Type::SRAM) {   // SRAM
		cell.heightInFeatureSize = param->heightInFeatureSizeSRAM;                   // Cell height in feature size
		cell.widthInFeatureSize = param->widthInFeatureSizeSRAM;                     // Cell width in feature size
		cell.widthSRAMCellNMOS = param->widthSRAMCellNMOS;
		cell.widthSRAMCellPMOS = param->widthSRAMCellPMOS;
		cell.widthAccessCMOS = param->widthAccessCMOS;
		cell.minSenseVoltage = param->minSenseVoltage;
	} else {
		cell.heightInFeatureSize = (cell.accessType==CMOS_access)? param->heightInFeatureSize1T1R : param->heightInFeatureSizeCrossbar;         // Cell height in feature size
		cell.widthInFeatureSize = (cell.accessType==CMOS_access)? param->widthInFeatureSize1T1R : param->widthInFeatureSizeCrossbar;            // Cell width in feature size
	}

	subArray->XNORparallelMode = param->XNORparallelMode;
	subArray->XNORsequentialMode = param->XNORsequentialMode;
	subArray->BNNparallelMode = param->BNNparallelMode;
	subArray->BNNsequentialMode = param->BNNsequentialMode;
	subArray->conventionalParallel = param->conventionalParallel;
	subArray->conventionalSequential = param->conventionalSequential;
	subArray->numRow = desiredIMCsize_x;
	subArray->numCol = desiredIMCsize_y;
	subArray->levelOutput = param->levelOutput;
	subArray->numWritePulse = param->numWritePulse;           // Only for memory mode (no trace-based)
	subArray->neuro = param->neuro;                           // Neuromorphic mode
	subArray->multifunctional = param->multifunctional;       // Multifunctional mode (not relevant for IMEC)
	subArray->parallelWrite = param->parallelWrite;           // Parallel write for crossbar RRAM in neuromorphic mode (not relevant for IMEC)
	subArray->numColMuxed = param->numColMuxed;               // How many columns share 1 read circuit (for neuro mode with analog RRAM) or 1 S/A (for memory mode or neuro mode with digital RRAM)
	subArray->numWriteColMuxed = param->numWriteColMuxed;     // How many columns share 1 write column decoder driver (for memory or neuro mode with digital RRAM)
    subArray->clkFreq = param->clkFreq;                       // Clock frequency
	subArray->relaxArrayCellHeight = param->relaxArrayCellHeight;
	subArray->relaxArrayCellWidth = param->relaxArrayCellWidth;
	subArray->numReadPulse = param->numBitInput;
	subArray->avgWeightBit = param->cellBit;
	subArray->numCellPerSynapse = param->numColPerSynapse;
	subArray->spikingMode = NONSPIKING;
	subArray->currentMode = param->currentMode;

	int numRow = desiredIMCsize_x;
	// int numCol = int(ceil(param->numColSubArray * param->sparsity));
	int numCol = desiredIMCsize_y;

	if (subArray->numColMuxed > numCol) {                      // Set the upperbound of numColMuxed
		subArray->numColMuxed = numCol;
	}

	subArray->numReadCellPerOperationFPGA = numCol;	           // Not relevant for IMEC
	subArray->numWriteCellPerOperationFPGA = numCol;	       // Not relevant for IMEC
	subArray->numReadCellPerOperationMemory = numCol;          // Define # of SRAM read cells in memory mode because SRAM does not have S/A sharing (not relevant for IMEC)
	subArray->numWriteCellPerOperationMemory = numCol/8;       // # of write cells per operation in SRAM memory or the memory mode of multifunctional memory (not relevant for IMEC)
	subArray->numReadCellPerOperationNeuro = numCol;           // # of SRAM read cells in neuromorphic mode
	subArray->numWriteCellPerOperationNeuro = numCol;	       // For SRAM or analog RRAM in neuro mode
    subArray->maxNumWritePulse = MAX(cell.maxNumLevelLTP, cell.maxNumLevelLTD);

	int numSubArrayRow = _numSubArrayRow;
	int numSubArrayCol = _numSubArrayCol;

	/*** initialize modules ***/
	subArray->Initialize(numRow, numCol, param->unitLengthWireResistance);        // initialize subArray
	if (param->parallelRead) {
		adderTree->Initialize(numSubArrayRow, log2((double)param->levelOutput)+param->numBitInput+param->numColPerSynapse+1, ceil((double)numSubArrayCol*(double)numCol/(double)param->numColMuxed));
	} else {
		adderTree->Initialize(numSubArrayRow, (log2((double)numRow)+param->cellBit-1)+param->numBitInput+param->numColPerSynapse+1, ceil((double)numSubArrayCol*(double)numCol/(double)param->numColMuxed));
	}

	bufferInput->Initialize(param->numBitInput*numRow, param->clkFreq);
	if (param->parallelRead) {
		bufferOutput->Initialize((numCol/param->numColMuxed)*(log2((double)param->levelOutput)+param->numBitInput+param->numColPerSynapse+adderTree->numStage), param->clkFreq);
	} else {
		bufferOutput->Initialize((numCol/param->numColMuxed)*((log2((double)numRow)+param->cellBit-1)+param->numBitInput+param->numColPerSynapse+adderTree->numStage), param->clkFreq);
	}

	subArray->CalculateArea();
	busInput->Initialize(HORIZONTAL, numSubArrayRow, numSubArrayCol, 0, numRow, subArray->height, subArray->width);
	busOutput->Initialize(VERTICAL, numSubArrayRow, numSubArrayCol, 0, numCol, subArray->height, subArray->width);
}


vector<double> ProcessingUnitCalculateArea(SubArray *subArray, int numSubArrayRow, int numSubArrayCol, double *height, double *width, double *bufferArea) {
	vector<double> areaResults;
	*height = 0;
	*width = 0;
	*bufferArea = 0;
	double area = 0;

	subArray->CalculateArea();

	adderTree->CalculateArea(NULL, subArray->width, NONE);
	bufferInput->CalculateArea(numSubArrayRow*subArray->height, NULL, NONE);
	bufferOutput->CalculateArea(NULL, numSubArrayCol*subArray->width, NONE);
	
	busInput->CalculateArea(1, true);
	busOutput->CalculateArea(1, true);

	// cout<<"\n*********************************************************************************\n";
	// cout<<"subArray->usedArea "<<subArray->usedArea<<endl;
	// cout<<"subArray->heightArray "<<subArray->heightArray<<endl;
	// cout<<"subArray->widthArray "<<subArray->widthArray<<endl;
	//cout<<"subArray->height "<<subArray->height<<endl;
	//cout<<"subArray->width "<<subArray->width<<endl;
	// cout<<"subArray->emptyArea "<<subArray->emptyArea<<endl;
	// cout<<"subArray->slSwitchMatrix.height"<<subArray->slSwitchMatrix.height<<endl;
	// cout<<"subArray->mux.height"<<subArray->mux.height<<endl;
	// cout<<"subArray->multilevelSenseAmp.height"<<subArray->multilevelSenseAmp.height<<endl;
	// cout<<"subArray->multilevelSAEncoder.height"<<subArray->multilevelSAEncoder.height<<endl;
	// cout<<"subArray->shiftAdd.height"<<subArray->shiftAdd.height<<endl;
	

	// cout<<"numSubArrayRow*numSubArrayCol "<<numSubArrayRow*numSubArrayCol<<endl;
	// cout<<"adderTree->area "<<adderTree->area<<endl;
	//cout<<"bufferInput->area "<<bufferInput->area<<endl;
	//cout<<"bufferOutput->area "<<bufferOutput->area<<endl;
	// cout<<"\n*********************************************************************************\n";

	area = subArray->usedArea * (numSubArrayRow*numSubArrayCol) + adderTree->area + bufferInput->area + bufferOutput->area;
	//cout<<"The subarray area is : "<<subArray->usedArea * (numSubArrayRow*numSubArrayCol)<<" um^2"<<endl;
	*height = sqrt(area);
	*width = area/(*height);

	areaResults.push_back(area);
	areaResults.push_back(subArray->areaADC*(numSubArrayRow*numSubArrayCol));
	areaResults.push_back(subArray->areaAccum*(numSubArrayRow*numSubArrayCol)+adderTree->area);
	areaResults.push_back(subArray->areaOther*(numSubArrayRow*numSubArrayCol));
	areaResults.push_back(bufferInput->area + bufferOutput->area);
	areaResults.push_back(subArray->areaOther*(numSubArrayRow*numSubArrayCol));
	areaResults.push_back(subArray->areaArray * (numSubArrayRow*numSubArrayCol));
	//cout<<"bufferInput->area"<<bufferInput->area <<endl;
	//cout<<"bufferOutput->area"<<bufferOutput->area<<endl;
	return areaResults;
}


double ProcessingUnitCalculatePerformance(SubArray *subArray, const vector<vector<double> > &newMemory, const vector<vector<double> > &oldMemory,
											const vector<vector<double> > &inputVector,
											int arrayDupRow, int arrayDupCol, int numSubArrayRow, int numSubArrayCol, int weightMatrixRow,
											int weightMatrixCol, int numInVector, MemCell& cell, double *readLatency, double *readDynamicEnergy, double *leakage,
											double *bufferLatency, double *bufferDynamicEnergy, double *icLatency, double *icDynamicEnergy,
											double *coreLatencyADC, double *coreLatencyAccum, double *coreLatencyOther, double *coreLatencyArray, double *coreEnergyADC,
											double *coreEnergyAccum, double *coreEnergyOther, double *coreEnergyArray, double IMC_size_x, double IMC_size_y) {

	/*** define how many subArray are used to map the whole layer ***/
	*readLatency = 0;
	*readDynamicEnergy = 0;
	*leakage = 0;
	*bufferLatency = 0;
	*bufferDynamicEnergy = 0;
	*icLatency = 0;
	*icDynamicEnergy = 0;
	*coreEnergyADC = 0;
	*coreEnergyAccum = 0;
	*coreEnergyOther = 0;
	*coreEnergyArray = 0;
	*coreLatencyADC = 0;
	*coreLatencyAccum = 0;
	*coreLatencyOther = 0;
	*coreLatencyArray = 0;
	
	double subArrayReadLatency = 0;
	double subArrayReadDynamicEnergy = 0;
	double subArrayLeakage = 0;
	double subArrayLatencyADC = 0;
	double subArrayLatencyAccum = 0;
	double subArrayLatencyOther = 0;
	double subArrayLatencyArray=0;
	//cout<<"arrayDupRow"<<arrayDupRow<<endl;
	//cout<<"arrayDupCol"<<arrayDupCol<<endl;
	//cout<<"weightMatrixRow"<<weightMatrixRow<<endl;
	//cout<<"weightMatrixCol"<<weightMatrixCol<<endl;
	if (arrayDupRow*arrayDupCol > 1) {
		//cout<<"\n Entered array DUp if in pU.cpp"<<endl;
		// weight matrix is duplicated among subArray
		if (arrayDupRow < numSubArrayRow || arrayDupCol < numSubArrayCol) {
			// a couple of subArrays are mapped by the matrix
			// need to redefine the data-grab start-point
			for (int i=0; i<ceil((double) weightMatrixRow/(double) IMC_size_x); i++) {
				for (int j=0; j<ceil((double) weightMatrixCol/(double) IMC_size_y); j++) {
					int numRowMatrix = min(IMC_size_x, weightMatrixRow-i*IMC_size_x);
					int numColMatrix = min(IMC_size_y, weightMatrixCol-j*IMC_size_y);
					
					if ((i*IMC_size_x < weightMatrixRow) && (j*IMC_size_y < weightMatrixCol) && (i*IMC_size_x < weightMatrixRow) ) {
						// assign weight and input to specific subArray
						vector<vector<double> > subArrayMemory;
						subArrayMemory = CopySubArray(newMemory, i*IMC_size_x, i*IMC_size_y, numRowMatrix, numColMatrix);
						vector<vector<double> > subArrayInput;
						subArrayInput = CopySubInput(inputVector, i*IMC_size_x, numInVector, numRowMatrix);

						for (int i=0; i<numInVector; i++) 
						{                 // calculate single subArray through the total input vectors
							double activityRowRead = 0;
							vector<double> input;
							input = GetInputVector(subArrayInput, i, &activityRowRead);
							subArray->activityRowRead = activityRowRead;

							int cellRange = pow(2, param->cellBit);
							if (param->parallelRead) {
								subArray->levelOutput = param->levelOutput;               // # of levels of the multilevelSenseAmp output
							} else {
								subArray->levelOutput = cellRange;
							}

							vector<double> columnResistance;
							columnResistance = GetColumnResistance(input, subArrayMemory, cell, param->parallelRead, subArray->resCellAccess);

							subArray->CalculateLatency(1e20, columnResistance);
							subArray->CalculatePower(columnResistance);

							subArrayReadLatency += subArray->readLatency;
							*readDynamicEnergy += subArray->readDynamicEnergy;
							subArrayLeakage = subArray->leakage;

							subArrayLatencyADC += subArray->readLatencyADC;
							subArrayLatencyAccum += subArray->readLatencyAccum;
							subArrayLatencyOther += subArray->readLatencyOther;
							
							*coreEnergyADC += subArray->readDynamicEnergyADC;
							*coreEnergyAccum += subArray->readDynamicEnergyAccum;
							*coreEnergyOther += subArray->readDynamicEnergyOther;
						}
						adderTree->CalculateLatency((int)(numInVector/param->numBitInput)*param->numColMuxed, ceil((double) weightMatrixRow/(double) IMC_size_x), 0);
						adderTree->CalculatePower((int)(numInVector/param->numBitInput)*param->numColMuxed, ceil((double) weightMatrixRow/(double) IMC_size_x));

						*readLatency = max(subArrayReadLatency + adderTree->readLatency, (*readLatency));
						*readDynamicEnergy += adderTree->readDynamicEnergy;

						*coreLatencyADC = MAX(subArrayLatencyADC, (*coreLatencyADC));
						//cout<<"\n With duplication is coreLatencyAccum is: "<<*coreLatencyAccum<<endl;
						//cout<<"\n The adder tree latency is: "<<adderTree->readLatency<<endl;
						*coreLatencyAccum = MAX(subArrayLatencyAccum + adderTree->readLatency, (*coreLatencyAccum));
						*coreLatencyOther = MAX(subArrayLatencyOther, (*coreLatencyOther));

						*coreEnergyAccum += adderTree->readDynamicEnergy;
					}
				}
			}
			// considering speedup, the latency of processing each layer is decreased

			*readLatency = (*readLatency)/(arrayDupRow*arrayDupCol);

			*coreLatencyADC = (*coreLatencyADC)/(arrayDupRow*arrayDupCol);
			*coreLatencyAccum = (*coreLatencyAccum)/(arrayDupRow*arrayDupCol);
			*coreLatencyOther = (*coreLatencyOther)/(arrayDupRow*arrayDupCol);

			//cout<<"latency"<<endl;
			//cout<<*readLatency<<endl;
			//cout<<*coreLatencyADC<<endl;
			//cout<<*coreLatencyAccum<<endl;
			//cout<<*coreLatencyOther<<endl;
			//return 0;
			//*readLatency += *coreLatencyADC + *coreLatencyAccum + *coreLatencyOther;
		}	else {
			// assign weight and input to specific subArray
			vector<vector<double> > subArrayMemory;
			subArrayMemory = CopySubArray(newMemory, 0, 0, weightMatrixRow, weightMatrixCol);
			vector<vector<double> > subArrayInput;
			subArrayInput = CopySubInput(inputVector, 0, numInVector, weightMatrixRow);

			for (int i=0; i<numInVector; i++) {                 // calculate single subArray through the total input vectors
				double activityRowRead = 0;
				vector<double> input;
				input = GetInputVector(subArrayInput, i, &activityRowRead);
				subArray->activityRowRead = activityRowRead;
				int cellRange = pow(2, param->cellBit);

				if (param->parallelRead) {
					subArray->levelOutput = param->levelOutput;               // # of levels of the multilevelSenseAmp output
				} else {
					subArray->levelOutput = cellRange;
				}

				vector<double> columnResistance;
				columnResistance = GetColumnResistance(input, subArrayMemory, cell, param->parallelRead, subArray->resCellAccess);

				subArray->CalculateLatency(1e20, columnResistance);
				subArray->CalculatePower(columnResistance);

				subArrayReadLatency += subArray->readLatency;
				*readDynamicEnergy += subArray->readDynamicEnergy;
				subArrayLeakage = subArray->leakage;

				subArrayLatencyADC += subArray->readLatencyADC;
				subArrayLatencyAccum += subArray->readLatencyAccum;
				subArrayLatencyOther += subArray->readLatencyOther;

				*coreEnergyADC += subArray->readDynamicEnergyADC;
				*coreEnergyAccum += subArray->readDynamicEnergyAccum;
				*coreEnergyOther += subArray->readDynamicEnergyOther;
			}
			// do not pass adderTree
			*readLatency = subArrayReadLatency/(arrayDupRow*arrayDupCol);
			*readDynamicEnergy += subArrayReadDynamicEnergy;

			*coreLatencyADC = subArrayLatencyADC/(arrayDupRow*arrayDupCol);

			*coreLatencyAccum = subArrayLatencyAccum/(arrayDupRow*arrayDupCol);
			*coreLatencyOther = subArrayLatencyOther/(arrayDupRow*arrayDupCol);

			//cout<<"latency"<<endl;
			//cout<<*readLatency<<endl;
			//cout<<*coreLatencyADC<<endl;
			//cout<<*coreLatencyAccum<<endl;
			//cout<<*coreLatencyOther<<endl;
			//return 0;
		}
	}


	else {
		//cout<<"\n Entered array DUp if in pU.cpp"<<endl;
		//weight matrix is further partitioned inside PE (among subArray) --> no duplicated
		for (int i=0; i<numSubArrayRow/*ceil((double) weightMatrixRow/(double) param->numRowSubArray)*/; i++) {
			for (int j=0; j<numSubArrayCol/*ceil((double) weightMatrixCol/(double) param->numColSubArray)*/; j++) {
				if ((i*IMC_size_x < weightMatrixRow) && (j*IMC_size_y < weightMatrixCol) && (i*IMC_size_x < weightMatrixRow) ) {
					int numRowMatrix = min(IMC_size_x, weightMatrixRow-i*IMC_size_x);
					int numColMatrix = min(IMC_size_y, weightMatrixCol-j*IMC_size_y);
					// assign weight and input to specific subArray
					vector<vector<double> > subArrayMemory;
					subArrayMemory = CopySubArray(newMemory, i*IMC_size_x, j*IMC_size_y, numRowMatrix, numColMatrix);
					vector<vector<double> > subArrayInput;
					subArrayInput = CopySubInput(inputVector, i*IMC_size_x, numInVector, numRowMatrix);
					
					subArrayReadLatency = 0;
					subArrayLatencyADC = 0;
					subArrayLatencyAccum = 0;
					subArrayLatencyOther = 0;
					subArrayLatencyArray =0;
					//cout<<"numInVector"<<numInVector<<endl;
					for (int i=0; i<numInVector; i++) {                 // calculate single subArray through the total input vectors
						double activityRowRead = 0;
						vector<double> input;
						input = GetInputVector(subArrayInput, i, &activityRowRead);
						subArray->activityRowRead = activityRowRead;

						int cellRange = pow(2, param->cellBit);
						if (param->parallelRead) {
							subArray->levelOutput = param->levelOutput;               // # of levels of the multilevelSenseAmp output
						} else {
							subArray->levelOutput = cellRange;
						}
						//cout<<"input"<<input[1]<<endl;
						vector<double> columnResistance;
						columnResistance = GetColumnResistance(input, subArrayMemory, cell, param->parallelRead, subArray->resCellAccess);
						
						subArray->CalculateLatency(1e20, columnResistance);
						subArray->CalculatePower(columnResistance);
						//cout<<"\n The ADC latency is: "<<subArray->readLatencyADC<<endl;
						subArrayReadLatency += subArray->readLatency;
						*readDynamicEnergy += subArray->readDynamicEnergy;
						subArrayLeakage = subArray->leakage;
						subArrayLatencyADC += subArray->readLatencyADC;
						subArrayLatencyAccum += subArray->readLatencyAccum;
						subArrayLatencyOther += subArray->readLatencyOther;
						subArrayLatencyArray += subArray->readLatencyArray;

						//cout<<"coreEnergyADC"<<*coreEnergyADC<<endl;
						//cout<<"numSubArrayCol"<<numSubArrayCol<<endl;
						//cout<<"numSubArrayRow"<<numSubArrayRow<<endl;
						*coreEnergyADC += subArray->readDynamicEnergyADC;
						*coreEnergyAccum += subArray->readDynamicEnergyAccum;
						*coreEnergyOther += subArray->readDynamicEnergyOther;
						*coreEnergyArray += subArray->readDynamicEnergyArray;
					}
					//cout<<"coreEnergyADC"<<*coreEnergyADC<<endl;
					//cout<<"subArray->readLatencyOther"<<subArray->readLatencyOther<<endl;
					adderTree->CalculateLatency((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse), ceil((double) weightMatrixRow/(double) IMC_size_x), 0);
					adderTree->CalculatePower((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse), ceil((double) weightMatrixRow/(double) IMC_size_x));

					*readLatency = max(subArrayReadLatency + adderTree->readLatency, (*readLatency));
					*readDynamicEnergy += adderTree->readDynamicEnergy;
					//cout<<"subArrayLatencyOther"<<subArrayLatencyOther<<endl;
					*coreLatencyADC = MAX(subArrayLatencyADC, (*coreLatencyADC));
					*coreLatencyAccum = MAX(subArrayLatencyAccum + adderTree->readLatency, (*coreLatencyAccum));
					//cout<<"subArrayLatencyAccum"<<subArrayLatencyAccum<<endl;
					//cout<<"adderTree->readLatency"<<adderTree->readLatency<<endl;
					//cout<<"coreLatencyAccum"<<*coreLatencyAccum<<endl;
					*coreLatencyOther = MAX(subArrayLatencyOther, (*coreLatencyOther));
					*coreLatencyArray = MAX(subArrayLatencyArray, (*coreLatencyArray));

					*coreEnergyAccum += adderTree->readDynamicEnergy;
				}
			}
		}
	}
	//considering buffer activation: no matter speedup or not, the total number of data transferred is fixed

	// input buffer: total num of data loaded in = weightMatrixRow*numInVector
	// output buffer: total num of data transferred = weightMatrixRow*numInVector/param->numBitInput (total num of IFM in the PE) *adderTree->numAdderTree*adderTree->numAdderBit (bit precision of OFMs)
	//Updated as per Neurosim 1.3
	/*
	bufferInput->CalculateLatency(0, numInVector);
	bufferOutput->CalculateLatency(0, numInVector/param->numBitInput);
	bufferInput->CalculatePower(weightMatrixRow, numInVector);
	bufferOutput->CalculatePower(weightMatrixCol*adderTree->numAdderBit/param->numBitInput, numInVector/param->numBitInput);
	*/
	bufferInput->CalculateLatency(0, weightMatrixRow/param->numRowPerSynapse*numInVector/(bufferInput->numDff));
	bufferOutput->CalculateLatency(0, weightMatrixCol/param->numColPerSynapse*adderTree->numAdderBit*numInVector/param->numBitInput/(bufferOutput->numDff));
	bufferInput->CalculatePower(weightMatrixRow/param->numRowPerSynapse*numInVector/(bufferInput->numDff), bufferInput->numDff);
	bufferOutput->CalculatePower(weightMatrixCol/param->numColPerSynapse*adderTree->numAdderBit*numInVector/param->numBitInput/(bufferOutput->numDff), bufferOutput->numDff);
			

	busInput->CalculateLatency(weightMatrixRow*numInVector/(busInput->numRow*busInput->busWidth));
	busOutput->CalculateLatency((weightMatrixCol*adderTree->numAdderBit*numInVector/param->numBitInput)/(busOutput->numRow*busOutput->busWidth));

	busInput->CalculatePower(busInput->numRow*busInput->busWidth, weightMatrixRow*numInVector/(busInput->numRow*busInput->busWidth));
	busOutput->CalculatePower(busOutput->numRow*busOutput->busWidth, (weightMatrixCol*adderTree->numAdderBit*numInVector/param->numBitInput)/(busOutput->numRow*busOutput->busWidth));

	*bufferLatency += bufferInput->readLatency + bufferOutput->readLatency;
	*icLatency += busInput->readLatency + busOutput->readLatency;
	//cout<<"*icLatency"<<*icLatency<<endl;
	// cout<<"\n*****************************************************************************\n";
	// cout<<"weightMatrixCol is "<<weightMatrixCol<<endl;
	// cout<<"weightMatrixRow is "<<weightMatrixRow<<endl;
	// cout<<"adderTree->numAdderBit is "<<adderTree->numAdderBit<<endl;
	// cout<<"param->numBitInput is "<<param->numBitInput<<endl;
	// cout<<"numInVector is "<<numInVector<<endl;
	// cout<<"bufferInput->readDynamicEnergy is "<<bufferInput->readDynamicEnergy<<endl;
	// cout<<"bufferOutput->readDynamicEnergy is "<<bufferOutput->readDynamicEnergy<<endl;
	// cout<<"Buffer->capTgDrain is "<<bufferInput->capTgDrain<<endl;
	// cout<<"Buffer->capTgGateN is "<<bufferInput->capTgGateN<<endl;
	// cout<<"Buffer->capTgGateP is "<<bufferInput->capTgGateP<<endl;
	// cout<<"Buffer->capTgGateN is "<<bufferInput->capTgGateN<<endl;
	// cout<<"Buffer->capInvInput is "<<bufferInput->capInvInput<<endl;
	// // cout<<"Buffer->numRead is "<<bufferInput->numRead<<endl;
	// // cout<<"Buffer->numDffPerOperation is "<<bufferInput->numDffPerOperation<<endl;
	// cout<<"Buffer->numDff is "<<bufferInput->numDff<<endl;
	// cout<<"\n*****************************************************************************\n";

	*bufferDynamicEnergy += bufferInput->readDynamicEnergy + bufferOutput->readDynamicEnergy;
	//cout<<"bufferDynamicEnergy_PE"<<*bufferDynamicEnergy<<endl;
	*icDynamicEnergy += busInput->readDynamicEnergy + busOutput->readDynamicEnergy;

	*readLatency += (*bufferLatency) + (*icLatency);
	*readDynamicEnergy += (*bufferDynamicEnergy) + (*icDynamicEnergy);
	*leakage = subArrayLeakage*numSubArrayRow*numSubArrayCol + adderTree->leakage + bufferInput->leakage + bufferOutput->leakage;
	//*coreLatencyOther += (*bufferLatency) + (*icLatency);
	//*coreEnergyOther += (*bufferDynamicEnergy) + (*icDynamicEnergy);

	// TODO: Training

}


vector<vector<double> > CopySubArray(const vector<vector<double> > &orginal, int positionRow, int positionCol, int numRow, int numCol) {
	vector<vector<double> > copy;
	for (int i=0; i<numRow; i++) {
		vector<double> copyRow;
		for (int j=0; j<numCol; j++) {
			copyRow.push_back(orginal[positionRow+i][positionCol+j]);
		}
		copy.push_back(copyRow);
		copyRow.clear();
	}
	return copy;
	copy.clear();
}


vector<vector<double> > CopySubInput(const vector<vector<double> > &orginal, int positionRow, int numInputVector, int numRow) {
	vector<vector<double> > copy;
	for (int i=0; i<numRow; i++) {
		vector<double> copyRow;
		for (int j=0; j<numInputVector; j++) {
			copyRow.push_back(orginal[positionRow+i][j]);
		}
		copy.push_back(copyRow);
		copyRow.clear();
	}
	return copy;
	copy.clear();
}


vector<double> GetInputVector(const vector<vector<double> > &input, int numInput, double *activityRowRead) {
	vector<double> copy;
	for (int i=0; i<input.size(); i++) {
		double x = input[i][numInput];
		copy.push_back(x);
	}
	double numofreadrow = 0;  // initialize readrowactivity parameters
	for (int i=0; i<input.size(); i++) {
		if (copy[i] != 0) {
			numofreadrow += 1;
		}else {
			numofreadrow += 0;
		}
	}
	double totalnumRow = input.size();
	*(activityRowRead) = numofreadrow/totalnumRow;
	return copy;
	copy.clear();
}


vector<double> GetColumnResistance(const vector<double> &input, const vector<vector<double> > &weight, MemCell& cell, bool parallelRead, double resCellAccess) {
	vector<double> resistance;
	vector<double> conductance;
	double columnG = 0;

	for (int j=0; j<weight[0].size(); j++) {
		int activatedRow = 0;
		columnG = 0;
		for (int i=0; i<weight.size(); i++) {
			if (cell.memCellType == Type::RRAM) {	// eNVM
				double totalWireResistance;
				if (cell.accessType == CMOS_access) {
					totalWireResistance = (double) 1.0/weight[i][j] + (j + 1) * param->wireResistanceRow + (weight.size() - i) * param->wireResistanceCol + cell.resistanceAccess;
				} else {
					totalWireResistance = (double) 1.0/weight[i][j] + (j + 1) * param->wireResistanceRow + (weight.size() - i) * param->wireResistanceCol;
				}
				if ((int) input[i] == 1) {
					columnG += (double) 1.0/totalWireResistance;
					activatedRow += 1 ;
				} else {
					columnG += 0;
				}
			} else if (cell.memCellType == Type::FeFET) {
				double totalWireResistance;
				totalWireResistance = (double) 1.0/weight[i][j] + (j + 1) * param->wireResistanceRow + (weight.size() - i) * param->wireResistanceCol;
				if ((int) input[i] == 1) {
					columnG += (double) 1.0/totalWireResistance;
					activatedRow += 1 ;
				} else {
					columnG += 0;
				}

			} else if (cell.memCellType == Type::SRAM) {
				// SRAM: weight value do not affect sense energy --> read energy calculated in subArray.cpp (based on wireRes wireCap etc)
				if ((int) input[i] == 1) {
					columnG += (double) 1.0/resCellAccess + (double) 1.0/param->wireResistanceCol;
					activatedRow += 1 ;
				} else {
					columnG += (double) 1.0/param->wireResistanceCol;
				}
			}
		}

		if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET) {
			if (!parallelRead) {
				conductance.push_back((double) columnG/activatedRow);
			} else {
				conductance.push_back(columnG);
			}
		} else {
			conductance.push_back(columnG);
		}
	}
	// covert conductance to resistance
	for (int i=0; i<weight[0].size(); i++) {
		resistance.push_back((double) 1.0/conductance[i]);
	}

	return resistance;
	resistance.clear();
}
