
# coding: utf-8

# In[28]:


# Chawan, Varsha Rani
# 1001-553-524
# 2018-09-23
# Assignment-02-01

import sys
if sys.version_info[0] < 3:
	import Tkinter as tk
else:
	import tkinter as tk

from tkinter import simpledialog
from tkinter import filedialog
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import seed
from numpy.random import randint
import time

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.backends.tkagg as tkagg
from matplotlib import colors as c


# In[29]:


class MainWindow(tk.Tk):
	"""
	This class creates and controls the main window frames and widgets
	Chawan Varsha Rani 2018_09_23
	"""

	def __init__(self, debug_print_flag=False):
		tk.Tk.__init__(self)
		self.debug_print_flag = debug_print_flag
		self.master_frame = tk.Frame(self)
		self.master_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
		# set the properties of the row and columns in the master frame
		self.rowconfigure(0, weight=1, minsize=500)
		self.columnconfigure(0, weight=1, minsize=500)
		self.master_frame.rowconfigure(2, weight=10, minsize=100, uniform='xx')
		self.master_frame.columnconfigure(0, weight=1, minsize=200, uniform='xx')
		# Create an object for plotting graphs in the left frame
		self.left_frame = tk.Frame(self.master_frame)
		self.left_frame.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
		self.display_decision_boundary = Perceptron(self, self.left_frame, debug_print_flag=self.debug_print_flag)


# In[30]:


class Perceptron:
	"""
	This class creates and controls the sliders , buttons , drop down in the frame which
	are used to display decision bounrdy and generate samples and train .
	"""

	def __init__(self, root, master, debug_print_flag=False):
		self.master = master
		self.root = root
		#########################################################################
		#  Set up the constants and default values
		#########################################################################
		self.xmin = -10
		self.xmax = 10
		self.ymin = -10
		self.ymax = 10
		self.input_weight1 = 1
		self.input_weight2 = 1
		self.bias = 0.0
		self.activation_type = "Symmetrical Hard limit"
		self.epoch = 100
		self.learning_rate = 0.05
		self.targets = np.array([-1,-1,1,1])
		self.sample_data = self.generate_random_samples()

		#########################################################################
		#  Set up the plotting frame and controls frame
		#########################################################################
		master.rowconfigure(0, weight=10, minsize=200)
		master.columnconfigure(0, weight=1)
		self.plot_frame = tk.Frame(self.master, borderwidth=10, relief=tk.SUNKEN)
		self.plot_frame.grid(row=0, column=0, columnspan=1, sticky=tk.N + tk.E + tk.S + tk.W)
		self.figure = plt.figure("")
		self.axes = self.figure.add_axes([0.15, 0.15, 0.6, 0.8])
		self.axes = self.figure.gca()
		self.axes.set_xlabel('Input1')
		self.axes.set_ylabel('Input2')
		# self.axes.margins(0.5)
		self.axes.set_title("")
		plt.xlim(self.xmin, self.xmax)
		plt.ylim(self.ymin, self.ymax)
		self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
		self.plot_widget = self.canvas.get_tk_widget().pack(side=tk.TOP , fill=tk.BOTH , expand=True)
		# Create a frame to contain all the controls such as sliders, buttons, ...
		self.controls_frame = tk.Frame(self.master)
		self.controls_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
		#########################################################################
		#  Set up the control widgets such as sliders ,buttons and dropdown 
		#########################################################################
		self.input_weight1_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
		                                    from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
		                                    activebackground="#FF0000", highlightcolor="#00FFFF", label="Weight W1",
		                                    command=lambda event: self.input_weight1_slider_callback())
		self.input_weight1_slider.set(self.input_weight1)
		self.input_weight1_slider.bind("<ButtonRelease-1>", lambda event: self.input_weight1_slider_callback())
		self.input_weight1_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
		self.input_weight2_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
		                                    from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
		                                    activebackground="#FF0000", highlightcolor="#00FFFF", label="Weight W2",
		                                    command=lambda event: self.input_weight2_slider_callback())
		self.input_weight2_slider.set(self.input_weight2)
		self.input_weight2_slider.bind("<ButtonRelease-2>", lambda event: self.input_weight2_slider_callback())
		self.input_weight2_slider.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)        
		self.bias_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL, from_=-10.0,
		                            to_=10.0, resolution=0.01, bg="#DDDDDD", activebackground="#FF0000",
		                            highlightcolor="#00FFFF", label="Bias",
		                            command=lambda event: self.bias_slider_callback())
		self.bias_slider.set(self.bias)
		self.bias_slider.bind("<ButtonRelease-1>", lambda event: self.bias_slider_callback())
		#self.bias_slider.bind("<ButtonRelease-2>", lambda event: self.bias_slider_callback())
		self.bias_slider.grid(row=0, column=2, sticky=tk.N + tk.E + tk.S + tk.W)
		self.train = tk.Button(self.controls_frame, text="Train", width=16,command=self.train_button_callback)
		self.train.grid(row=0, column=3)
		self.randomData = tk.Button(self.controls_frame, text="Create Samples", width=16,command=self.randomData_button_callback)
		self.randomData.grid(row=0, column=4)
        ######################################################################### 
		#  Set up the frame for drop down selection
		#########################################################################
		self.label_for_activation_function = tk.Label(self.controls_frame, text="Activation Function Type:",
		                                              justify="center")
		self.label_for_activation_function.grid(row=0, column=5, sticky=tk.N + tk.E + tk.S + tk.W)
		self.activation_function_variable = tk.StringVar()
		self.activation_function_dropdown = tk.OptionMenu(self.controls_frame, self.activation_function_variable,
		                                                  "Symmetrical Hard limit", "Hyperbolic Tangent","Linear", command=lambda
				event: self.activation_function_dropdown_callback())
		self.activation_function_variable.set("Symmetrical Hard limit")
		self.activation_function_dropdown.grid(row=0, column=6, sticky=tk.N + tk.E + tk.S + tk.W)

	def generate_random_samples(self):
        ######################################################################### 
		#  Generate Random Samples either using randomint and concatenating for 4 points
        #  or using numpy.multivariate random variable
		#########################################################################
		a = np.random.randint(-10,10, size=(1, 2))
		b = np.random.randint(-10,10, size=(1, 2))
		c = np.random.randint(-10,10, size=(1, 2))
		d = np.random.randint(-10,10, size=(1, 2))

		e = np.concatenate((a,b),axis = 0)
		f = np.concatenate((e,c),axis = 0)
		sample_data = np.concatenate((f,d),axis = 0)  
		self.sample_data = sample_data
		return sample_data
        
        
	def plot_decision_boundary(self,w1,w2):
        ######################################################################### 
		#  Plot decision boundry 
		#########################################################################
		self.axes.cla()
		resolution=100
		xs = np.linspace(self.xmin, self.xmax, resolution)
		ys = np.linspace(self.ymin, self.ymax, resolution)
		xx, yy = np.meshgrid( xs,ys)
		zz = w1*xx+w2*yy + self.bias
		zz[zz>0]=+1
		zz[zz<0]=-1
		cMap = c.ListedColormap(['r','g'])
		self.axes.pcolormesh(xx, yy, zz,cmap=cMap)
		self.axes.set_xlabel('Input1')
		self.axes.set_ylabel('Input2')
		for sample ,target in zip(self.sample_data,self.targets):
			plt.plot(sample[0],sample[1],'bo' if (target == 1) else 'wo')
		self.axes.xaxis.set_visible(True)
		plt.xlim(self.xmin, self.xmax)
		plt.ylim(self.ymin, self.ymax)
		plt.title(self.activation_type)
		self.canvas.draw() 

	def calculate_activation_function(self,bias,sample,weight,type='Symmetrical Hard limit'): 
        ######################################################################### 
		#  Calculates the actual value
		#########################################################################
		weightT = np.transpose(weight)
		net_value = np.dot(sample, weightT) + bias  
		if type == 'Symmetrical Hard limit':           
			if net_value >= 0.0 :
				actual_output = 1 
			elif net_value < 0.0 :
				actual_output = -1 
		elif type == "Hyperbolic Tangent":
			actual_output = (np.exp(net_value)-np.exp(-net_value)) / (np.exp(net_value)+np.exp(-net_value))
		elif type == "Linear":
			actual_output = net_value            
		return actual_output


        
	def train_perceptron(self,sample_data,targets,W1,W2): 
        ######################################################################### 
		#  Trains the weights and bias as per perceptron learning rule
		#########################################################################
		self.weight= np.ones((1,2))
		self.weight[0,0] = W1
		self.weight[0,1] = W2
		for e in range(self.epoch):
			for sample, target in zip(sample_data,targets):  
				actual_output = self.calculate_activation_function(self.bias,sample,self.weight,self.activation_type)
				self.error = (target - actual_output)
				# Keeping the error in range from -700 to 700 this is to avoid nan or overflow of weight in case of linear function
				if self.error> 700 or self.error < -700:
					self.error /= 10000
				self.weight += self.learning_rate * self.error  * sample
				self.bias += self.learning_rate * self.error 
		W1 = self.weight[0,0]
		W2 = self.weight[0,1]  
		self.plot_decision_boundary(W1,W2)

	def input_weight1_slider_callback(self):
		self.input_weight1 = np.float(self.input_weight1_slider.get())
		self.plot_decision_boundary(self.input_weight1,self.input_weight2)
        
	def input_weight2_slider_callback(self):
		self.input_weight2 = np.float(self.input_weight2_slider.get())
		self.plot_decision_boundary(self.input_weight1,self.input_weight2)

	def bias_slider_callback(self):
		self.bias = np.float(self.bias_slider.get())
		self.plot_decision_boundary(self.input_weight1,self.input_weight2)

	def activation_function_dropdown_callback(self):
		self.activation_type = self.activation_function_variable.get()
		self.train_perceptron(self.sample_data,self.targets,self.input_weight1,self.input_weight2)

	def randomData_button_callback(self):
		self.generate_random_samples()        
		self.plot_decision_boundary(self.input_weight1,self.input_weight2)
        
	def train_button_callback(self):
		self.train_perceptron(self.sample_data,self.targets,self.input_weight1,self.input_weight2)

def close_window_callback(root):
	if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
		root.destroy()


# In[31]:


main_window = MainWindow(debug_print_flag=False)


# In[32]:


main_window.wm_state('zoomed')


# In[33]:


main_window.title('Assignment_02 --  Chawan')


# In[34]:


main_window.minsize(800, 600)


# In[35]:


main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))


# In[ ]:


main_window.mainloop()

