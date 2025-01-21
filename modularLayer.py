import numpy as np
import pennylane as qml
import torch 
import hashlib

from function_library import binaryAmplitude

torch.set_default_dtype(torch.float64)
torch.set_default_device('cpu')

class PreProcessingLayer(torch.nn.Module):
    def __init__(self, qc, prec):
        super(PreProcessingLayer).__init__()
        self.qc = qc
        self.prec = prec
        self.lamb = binaryAmplitude
        
    def forward(self, x):
        if self.prec == 1:
            return 2*torch.asin(torch.sqrt(x))
        else:           
            return self.lambd(x,self.qc,self.prec)        
        
class PostProcessingLayer(torch.nn.Module):
    def __init__(self, qc, prec, factors = 1):
        super(PostProcessingLayer, self).__init__()
        
        self.factors = factors
        
        self.qc = qc
        self.prec = prec

    def forward(self, x):        
        output = (2**self.qc)*torch.matmul(x,self.factors)
        return output

class QNN:
    def __init__(self,device = 'default.qubit',D = 2,Q = 9, binary_precision = 1):
        self.Q = Q
        self.qc = int(np.ceil(np.log2(Q)))
        self.prec = binary_precision
        
        self.qubits = self.qc+self.prec
        
        self.indexWires = [q for q in range(self.qc)]
        self.valueWires = [q for q in range(self.qc,self.qc+self.prec)]
        
        self.ancillas = 0
        
        self.ancillaWires = []
        self.indexAncilla = None
        self.valueAncilla = None
        
        self.actions = 0
        
        self.weight_shapes = None
        self.shapes = []
        self.labels = []
        
        self.circ = []
        self.circuit = None
        self.model = None
        self.device = device

        self.string = []

    def passAsString(self, arr):
        string = ", ".join([str(item) for item in arr])
        string = "["+string+"]"
        return string
        
    def setAncilla(self, index_ancilla, value_ancilla):
        
        if index_ancilla:
            if self.indexAncilla is None:
                self.indexAncilla = True
                self.qubits += 1
                self.ancillas += 1
                self.ancillaWires = [q for q in range(self.qc+self.prec,self.qc+self.prec+self.ancillas)]
        
        elif value_ancilla:
            if self.valueAncilla is None:
                self.valueAncilla = True
                self.qubits += 1
                self.ancillas += 1
                self.ancillaWires = [q for q in range(self.qc+self.prec,self.qc+self.prec+self.ancillas)]
    
    def setWires(self,  wire_label, index_ancilla = False, value_ancilla = False):
        if wire_label == 'index':
            wires = 1*self.indexWires
            if index_ancilla:
                wires.append(self.ancillaWires[0])
        
        if wire_label == 'value':
            wires =  1*self.valueWires
            if value_ancilla:
                wires.append(self.ancillaWires[-1])
        
        elif wire_label == 'all':
            wires = [q for q in range(self.qubits)]

        return wires
        
    def addEntanglingLayer(self,depth,wire_label,strong = True, gateset = 'RY',index_ancilla = False, value_ancilla = False):
        self.string.append("E"+str(depth)+wire_label+str(int(strong))+gateset+str(int(index_ancilla))+str(int(value_ancilla)))
        self.setAncilla(index_ancilla,value_ancilla)
        wires = self.setWires(wire_label, index_ancilla, value_ancilla)
        
        self.labels += ["action"+str(self.actions)]
        
        if strong:
            self.shapes.append((depth,len(wires),3))
            self.circ.append("qml.StronglyEntanglingLayers(kwargs['"+self.labels[-1]+"'],wires = "+str(wires)+")")
        
        else:
            self.shapes.append((depth,len(wires)))
            self.circ.append("qml.BasicEntanglerLayers(kwargs['"+self.labels[-1]+"'],wires ="+str(wires)+", rotation = qml."+gateset+")")
        
        self.actions += 1
        
    def performControlledRotation(self,strong = True, gateset = 'RY',index_ancilla = False, value_ancilla = False, count = 0):
        self.string.append("CR"+str(int(strong))+gateset+str(int(index_ancilla))+str(int(value_ancilla))+str(int(count)))
        self.setAncilla(index_ancilla,value_ancilla)
        
        control_label = 'index'
        control_wires = self.setWires(control_label, index_ancilla, value_ancilla)
        
        target_label = 'value'
        target_wires = self.setWires(target_label, index_ancilla, value_ancilla)
        
        if count == 0:
            count = 2**len(control_wires) # or Q
        
        self.labels += ["action"+str(self.actions)]

        if strong:
            self.shapes.append((count,len(target_wires),3))
            for q in range(count):
                st = [int(s) for s in np.binary_repr(q,len(control_wires))]
                for b in range(len(target_wires)):
                    self.circ.append("qml.ctrl(qml.U3, control = "+str(control_wires)+", control_values = "+str(st)+")(*kwargs['"+self.labels[-1]+"']["+str(q)+"]["+str(b)+"] , wires = "+str(target_wires[b])+")")
        else:
            self.shapes.append((count,len(target_wires)))
            for q in range(count):
                st = [int(s) for s in np.binary_repr(q,len(control_wires))]
                for b in range(len(target_wires)):
                    self.circ.append("qml.ctrl(qml."+str(gateset)+", control = "+str(control_wires)+", control_values = "+str(st)+")(kwargs['"+self.labels[-1]+"']["+str(q)+"]["+str(b)+"],wires = "+str(target_wires[b])+")")
        
        self.actions += 1
        
    def measureAndResetAncillas(self):
        self.string.append("M")
        if self.ancillas > 0:
            self.circ.append("qml.measure(wires = "+str(self.ancillaWires)+", reset = True)")
    
    def adjointNActions(self,N,first = 0):
        self.string.append("A"+str(N)+"-"+str(first))
        for n in reversed(range(first,N)):
            self.circ.append("qml.adjoint("+self.circ[n]+")")
    
    def setDevice(self,device):
        self.string.insert(0,device)
        self.device = device

    def compileCircuit(self,skip_initialization = None):
        
        dev = qml.device(self.device,wires = self.qubits)
        
        @qml.qnode(dev)
        def circuit(inputs,**kwargs):
            # Feature Map
            if skip_initialization is None:
                for q in self.indexWires:
                    qml.Hadamard(wires = q)
                
                if self.prec == 1:
                    if (len(inputs.shape) == 2):
                        for q in range(inputs.shape[1]):
                            st = [int(s) for s in np.binary_repr(q,len(self.indexWires))]
                            for b in self.valueWires:
                                qml.ctrl(qml.RY,self.indexWires,st)(inputs[:,q],wires = b)
                    elif (len(inputs.shape) == 1):
                        for q in range(inputs.shape[0]):
                            st = [int(s) for s in np.binary_repr(q,len(self.indexWires))]
                            for b in self.valueWires:
                                qml.ctrl(qml.RY,self.indexWires,st)(inputs[q],wires = b)
                
                else:
                    qml.AmplitudeEmbedding(features = inputs
                                         , wires = range(self.qc+self.prec)
                                         , pad_with = 0
                                         , normalize = True
                                          )
            
            # Execute dynamically created code
            for line in self.circ:
                exec(line,globals(),locals())
                
            return qml.probs(wires = self.indexWires+self.valueWires)
        
        self.circuit = circuit
        
        return self.circuit
    
    def compileLayer(self):
        self.compileCircuit()
        weight_shapes = dict(zip(self.labels,self.shapes))
        # Wrap the quantum circuit in a Torch layer
        self.layer = qml.qnn.TorchLayer(self.circuit, weight_shapes)
        return self.layer

    # Pre-calculate the factors for binary conversion with a given binary precision
    def substringConversion(self):
        size = 2**(self.qc+self.prec)
        factors = torch.zeros(size = (size,self.Q), device = "cpu")
        
        for s in range(size):
            st = np.binary_repr(s,self.qc+self.prec)
            q = int(st[:-self.prec],2)
            if q < self.Q:
                if self.prec == 1:
                    factors[s,q] = int(st[self.qc:],2)
                else:
                    factors[s,q] = int(st[self.qc:],2)/2**self.prec

        return factors
    
    def compileModel(self):
            
        num_basis = 2**self.qc
        num_rep = 2**self.prec
    
        factors = self.substringConversion()
        
        qlayer = self.compileLayer()
        layers  = [PreProcessingLayer(self.qc,self.prec),qlayer,PostProcessingLayer(factors,self.qc,self.prec)]
        self.model = torch.nn.Sequential(*layers)
        self.md5 = hashlib.md5(' \n'.join(self.string).encode()).hexdigest()
        return self.model, ' \n'.join(self.circ),' \n'.join(self.string),self.md5
    