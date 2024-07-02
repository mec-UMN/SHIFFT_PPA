/*******************************************************************************
* Copyright (c) 2017-2020
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Yu Cao
* All rights reserved.
*
* This source code is part of SIAM - a device-circuit-algorithm framework to benchmark
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
#include "math.h"
#include "Param.h"

Param::Param() {

	operationmode = 2;     		// 1: conventionalSequential (Use several multi-bit RRAM as one synapse)
								// 2: conventionalParallel (Use several multi-bit RRAM as one synapse)

	memcelltype = 2;        	// 1: cell.memCellType = Type::SRAM
								// 2: cell.memCellType = Type::RRAM
								// 3: cell.memCellType = Type::FeFET

	accesstype = 4;         	// 1: cell.accessType = CMOS_access
								// 2: cell.accessType = BJT_access
								// 3: cell.accessType = diode_access
								// 4: cell.accessType = none_access (Crossbar Array)

	transistortype = 1;     	// 1: inputParameter.transistorType = conventional
								// 2: inputParameter.transistorType = FET_2D
								// 3: inputParameter.transistorType = TFET

	deviceroadmap = 1;      	// 1: inputParameter.deviceRoadmap = HP (high performance)
								// 2: inputParameter.deviceRoadmap = LSTP  (low standby power)

	globalBufferType = false;   // false: register file
								// true: SRAM
	globalBufferCoreSizeRow = 128;
	globalBufferCoreSizeCol = 128;

	tileBufferType = false;     // false: register file
								// true: SRAM

	//Updated as per Neurosim 1.3
	tileBufferCoreSizeRow = 32;
	tileBufferCoreSizeCol = 32;

	peBufferType = false;       // false: register file
								// true: SRAM

	chipActivation = true;      // false: activation (reLu/sigmoid) inside Tile
								// true: activation outside Tile

	reLu = true;                // false: sigmoid
								// true: reLu

	novelMapping = false;       // false: conventional mapping. We will add additional mapping strategies in coming versions.
	currentMode = true; 
	inputdacmode=true; 			//true:dac
								//false: switch matrix
	numRowSubArray = 64;       // # of rows in single subArray
	numColSubArray = 64;       // # of columns in single subArray

	numRowSubArray_little = 64;       // # of rows in single subArray for little core
	numColSubArray_little = 64;       // # of columns in single subArray for little core

	numRowSubArray_mid = 64;       // # of rows in single subArray for little core
	numColSubArray_mid = 64;       // # of columns in single subArray for little core

	numRowSubArray_big = 64;       // # of rows in single subArray for big core
	numColSubArray_big = 64;       // # of columns in single subArray for big core

	numRowSubArray_type1 = 1024;       // # of rows in single subArray for different type of chiplet
	numColSubArray_type1 = 1024;       // # of columns in single subArray for different type of chiplet

	numRowSubArray_type2 = 64;       // # of rows in single subArray for different type of chiplet
	numColSubArray_type2 = 64;       // # of columns in single subArray for different type of chiplet

	numRowSubArray_type3 = 64;       // # of rows in single subArray for different type of chiplet
	numColSubArray_type3 = 64;       // # of columns in single subArray for different type of chiplet

	numRowSubArray_type4 = 64;       // # of rows in single subArray for different type of chiplet
	numColSubArray_type4 = 64;       // # of columns in single subArray for different type of chiplet

	numRowSubArray_type5 = 64;       // # of rows in single subArray for different type of chiplet
	numColSubArray_type5 = 64;       // # of columns in single subArray for different type of chiplet

	numRowSubArray_type6 = 64;       // # of rows in single subArray for different type of chiplet
	numColSubArray_type6 = 64;       // # of columns in single subArray for different type of chiplet


	sparsity = 1;				// How much Coloumn sparsity you have. How many xnumRowSubArray are you skipping. Expandede to layer-wise sparsity. Not used now

	algoWeightMax = 1;
	algoWeightMin = -1;

	row_multiplier = 4;			// N Rows of xBar inside Tile
	col_multiplier = 4;			// M Col of xBar inside Tile	
	to_subtract = 0;			// Need to change and add specific subtract for row and col

	pool_stride = 2;
	
	//*****************************
	//Chiplet Additions
	mode = true; 				// true: Chiplet Activated
								// false: Regular Single Chip Mode

	chiplet_structure = true;	// true: All chiplets are of same size with size_chiplet number of CE's. (Generic Design)
								// false: All chiplets of different size depending on the weights to be mapped (Full custom design)
	size_chiplet_little = 36; 			// Gives the number of Tiles in the each chiplet. We assume all the chiplets are same size. Each tile has 16 crossbar arrays
	
	size_chiplet_mid =36;
	size_chiplet_big = 36; 			// Gives the number of Tiles in the each chiplet. We assume all the chiplets are same size. Each tile has 16 crossbar arrays

	size_chiplet_1 = 1; 			// Gives the number of Tiles in the each chiplet. We assume all the chiplets are same size. Each tile has 16 crossbar arrays
	
	size_chiplet_2 = 36;
	size_chiplet_3 = 36; 			// Gives the number of Tiles in the each chiplet. We assume all the chiplets are same size. Each tile has 16 crossbar arrays

	size_chiplet_4 = 36; 			// Gives the number of Tiles in the each chiplet. We assume all the chiplets are same size. Each tile has 16 crossbar arrays
	
	size_chiplet_5 = 36;
	size_chiplet_6 = 36; 			// Gives the number of Tiles in the each chiplet. We assume all the chiplets are same size. Each tile has 16 crossbar arrays
	
	size_chiplet = 36; 			// Gives the number of Tiles in the each chiplet. We assume all the chiplets are same size. Each tile has 16 crossbar arrays
	fixed_cnt_chiplet = false;	// true: We have a fixed number of chiplets in the fabric irrespective the number we need.
								// false: The required number of chiplets only exist. (Custom Design)
	Chiplet_GAccumulator = 32;	// User input to give the size of the parallel adds we do

	max_little_count = 100;
	max_big_count	 = 100;
	max_mid_count	 = 100;
    cnt_chiplet = max_little_count + max_big_count+ max_mid_count; 			// Optional input to get a fixed size. Need to incorporate
	//*****************************

	heightInFeatureSizeSRAM = 8;                  // SRAM Cell height in feature size
	widthInFeatureSizeSRAM = 20;                     // SRAM Cell width in feature size
	heightInFeatureSizeCrossbar = 2;                // RRAM Crossbar Cell height in feature size
	widthInFeatureSizeCrossbar = 2;                 // RRAM Crossbar Cell width in feature size

	widthSRAMCellNMOS = 2.08;
	widthSRAMCellPMOS = 1.23;
	widthAccessCMOS = 1.31;
	minSenseVoltage = 0.1;

	heightInFeatureSize1T1R = 4;                    // 1T1R Cell height in feature size
	widthInFeatureSize1T1R = 4;                     // 1T1R Cell width in feature size
	heightInFeatureSizeCrossbar = 2;                // RRAM Crossbar Cell height in feature size
	widthInFeatureSizeCrossbar = 2;                 // RRAM Crossbar Cell width in feature size

	relaxArrayCellHeight = 0;         // relax ArrayCellHeight or not
	relaxArrayCellWidth = 0;          // relax ArrayCellWidth or not

	globalBusDelayTolerance = 0.1;
	localBusDelayTolerance = 0.1;
	treeFoldedRatio = 4;
	maxGlobalBusWidth = 65536 ;    		// the max buswidth allowed on top-level
	clkFreq = 1e9;               // Clock frequency
	featuresize = 40e-9;         // Wire width for subArray simulation
	temp = 301;                  // Temperature (K)
	technode = 45;               // Technology
	wireWidth = 40;                                    // wireWidth of the cell for Accuracy calculation
	readNoise = 0.15;	                               // Sigma of read noise in gaussian distribution
	resistanceOn = 100e3;        // Ron resistance at Vr in the reported measurement data (need to recalculate below if considering the nonlinearity)
	resistanceOff = 10e6;        // Roff resistance at Vr in the reported measurement dat (need to recalculate below if considering the nonlinearity)
	maxConductance = (double) 1/resistanceOn;
	minConductance = (double) 1/resistanceOff;
	maxNumLevelLTP = 97;	                            // Maximum number of conductance states during LTP or weight increase
	maxNumLevelLTD = 100;	                            // Maximum number of conductance states during LTD or weight decrease
	readVoltage = 0.9;	                                // On-chip read voltage for memory cell
	readPulseWidth = 10e-9;
	accessVoltage = 1.1;                                       // Gate voltage for the transistor in 1T1R
	resistanceAccess = 15e3;
	multipleCells = 1;                                         // Value should be N^2 such as 1, 4, 9 ...etc
	nonlinearIV = 0;                                           // This option is to consider I-V nonlinearity in cross-point array or not
	nonlinearity = 10;                                         // This is the nonlinearity for the current ratio at Vw and Vw/2
	writeVoltage = 100e-9;
	writePulseWidth = 2;
	numWritePulse = 1;           // Only for memory mode (no trace-based)

	neuro = 1;                   // Neuromorphic mode
	multifunctional = 0;         // Multifunctional mode (not relevant for IMEC)
	parallelWrite = 0;           // Parallel write for crossbar RRAM in neuromorphic mode (not relevant for IMEC)
	numlut = 32;                 // # of LUT (not relevant for IMEC)
	numColMuxed = 8;             // How many columns share 1 read circuit (for neuro mode with analog RRAM) or 1 S/A (for memory mode or neuro mode with digital RRAM)
	numWriteColMuxed = 4;        // How many columns share 1 write column decoder driver (for memory or neuro mode with digital RRAM)
	levelOutput = 128;            // # of levels of the multilevelSenseAmp output or ADC
	cellBit =4;                 // precision of memory device
	
	if (memcelltype == 1) {
		cellBit = 2;             // force cellBit = 2 for all SRAM cases
	}

	/*** initialize operationMode as default ***/
	conventionalParallel = 0;
	conventionalSequential = 0;
	switch(operationmode) {
		case 2:	    conventionalParallel = 1;           break;
		case 1:	    conventionalSequential = 1;         break;
		case -1:	break;
		default:	exit(-1);
	}

	/*** parallel read ***/
	parallelRead = 0;
	if(conventionalParallel) {
		parallelRead = 1;
	} else {
		parallelRead = 0;
	}

	/*** Initialize interconnect wires ***/
	switch(wireWidth) {
		case 200: 	AR = 2.10; Rho = 2.42e-8; break;
		case 100:	AR = 2.30; Rho = 2.73e-8; break;
		case 50:	AR = 2.34; Rho = 3.91e-8; break;
		case 40:	AR = 1.90; Rho = 4.03e-8; break;
		case 32:	AR = 1.90; Rho = 4.51e-8; break;
		case 22:	AR = 2.00; Rho = 5.41e-8; break;
		case 14:	AR = 2.10; Rho = 7.43e-8; break;
		case -1:	break;	// Ignore wire resistance or user define
		default:	exit(-1); puts("Wire width out of range");
	}

	if (memcelltype == 1) {
		wireLengthRow = wireWidth * 1e-9 * heightInFeatureSizeSRAM;
		wireLengthCol = wireWidth * 1e-9 * widthInFeatureSizeSRAM;
	} else {
		if (accesstype == 1) {
			wireLengthRow = wireWidth * 1e-9 * heightInFeatureSize1T1R;
			wireLengthCol = wireWidth * 1e-9 * widthInFeatureSize1T1R;
		} else {
			wireLengthRow = wireWidth * 1e-9 * heightInFeatureSizeCrossbar;
			wireLengthCol = wireWidth * 1e-9 * widthInFeatureSizeCrossbar;
		}
	}

	if (wireWidth == -1) {
		unitLengthWireResistance = 1.0;	
		wireResistanceRow = 0;
		wireResistanceCol = 0;
	} else {
		unitLengthWireResistance =  Rho / ( wireWidth*1e-9 * wireWidth*1e-9 * AR );
		wireResistanceRow = unitLengthWireResistance * wireLengthRow;
		wireResistanceCol = unitLengthWireResistance * wireLengthCol;
	}

}
