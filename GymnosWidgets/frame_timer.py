#!/usr/bin/python

try:
    import tkinter
    from tkinter import *
except ImportError:
    import Tkinter as tkinter
    from tkinter import *

WIDTH = 256     #width of the window
HEIGHT = 286    #height of the window
OFFSET_X = 50   #offset for window x axis
OFFSET_Y = 0    #offset for window y axis
FONT_SIZE = 36

class frameTimers:

    def __init__(self, master):
        master.title("Frame Timers")
        master.geometry("%dx%d+%d+%d" % (WIDTH, HEIGHT, OFFSET_X, OFFSET_Y))
        self.canvas = Canvas(master, width=256, height=256, relief='raised', borderwidth=1)
        self.canvas.pack()

        self.rectangleOne = self.canvas.create_rectangle(0, 0, 126, 255, fill='red')
        self.labelOne = self.canvas.create_text(60, 20, font=("Purisa", FONT_SIZE),text="Squat")
        self.rectangleTwo = self.canvas.create_rectangle(126, 0, 255, 255, fill='blue')
        self.labelTwo = self.canvas.create_text(190, 20, font=("Purisa", FONT_SIZE),text="Bench")
        #self.labelTwo.pack(side=RIGHT, fill=BOTH, expand=TRUE)

        self.squat = self.canvas.create_text(55, 165, font=("Purisa", FONT_SIZE), text=0)
        self.bench = self.canvas.create_text(185, 165, font=("Purisa", FONT_SIZE), text=0)

        self.resetButton = Button(master, text="Reset Timers", command=self.resetTimers)
        self.resetButton.pack(side=BOTTOM)

    def squatTime(self, x):
        self.canvas.itemconfig(self.squat, text=x)

    def benchTime(self, x):
        self.canvas.itemconfig(self.bench, text=x)


    def resetTimers(self):
        self.squatTime(0)
        self.benchTime(0)

#ft.squatTime(100)    #this is how you use the class
#ft.benchTime(50)     #this is how you use the class