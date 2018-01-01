
import matplotlib.pyplot as plt
f = open("weight.txt","r") #opens file with name of "test.txt"

myList = []

for line in f:
    x = float(line)

    myList.append(x)

#print(myList)

weights = myList

max_val = max(weights)
min_val = min(weights)
print(max_val, " ",min_val)

size_weights=len(weights)
x_division = 13

graph_plot = []
for i in range(x_division):
    graph_plot.append(0)

'''
for i in range(size_weights):
    if(weights[i]> 0.015):
        graph_plot[0]+=1
    elif(weights[i]<= 0.015 and weights[i]>0):
        graph_plot[1] += 1
    elif (weights[i] <= 0 and weights[i] > -0.015):
        graph_plot[2] += 1
    elif (weights[i] <= -0.015):
        graph_plot[3] += 1
'''

x_axis=[i for i in range(x_division)]

difference = max_val - min_val
step_size = difference/x_division
#print(difference)

for i in range(size_weights):
    if(weights[i]>= 9*step_size+min_val):
        graph_plot[x_division-1] += 1

for i in range(size_weights):
    if(weights[i]< step_size+min_val):
        graph_plot[0] += 1


for i in range(x_division-2):
    for j in range(size_weights):
        if((weights[j]>= (min_val + (i+1)*step_size)) and (weights[j]<(min_val + (i+2)*step_size))):
            graph_plot[i+1] +=1



print(graph_plot)

for i in range(len(graph_plot)):
    x = graph_plot[i]
    graph_plot[i] = (x/size_weights)*100

print(graph_plot)

f.close()

thefile = open('plotSpec.txt', 'w')
thefile.write("max: %s\n" % max_val)
thefile.write("min: %s\n" % min_val)

for g in graph_plot:
    thefile.write("%s " % g)
thefile.close()
# plotting the points
plt.plot(x_axis, graph_plot)

# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')

# giving a title to my graph
plt.title('My first graph!')

# function to show the plot
plt.show()
