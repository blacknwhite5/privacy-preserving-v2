import matplotlib.pyplot as plt
import numpy as np

def parse_log(logfile):
    with open(logfile, 'r') as f:
        data = f.read()

    epoch, loss_d, loss_g = [], [], []
    for i, line in enumerate(data.split('\n')[1:-1]):
        line = line.split()
        
#        ep = int(line[0].split(':')[1])
        ep = 20000 * i
        l_d = float(line[2].split(':')[1])
        l_g = float(line[3].split(':')[1])
        epoch.append(ep)
        loss_d.append(l_d)
        loss_g.append(l_g)

    return epoch, loss_d, loss_g

def draw_results(epoch, losses_D, losses_G, show=False):
    axes_cycle = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] 
    modula = len(axes_cycle) 
    
    plt.figure() 
    plt.plot(epoch, losses_D, color=axes_cycle[0], label='loss_D') 
    plt.plot(epoch, losses_G, color=axes_cycle[2], label='loss_G') 
#    plt.plot(epoch[np.argmin(losses_D)], min(losses_D), 'o', color=axes_cycle[3], label='min D loss') 
#    plt.plot(epoch[np.argmin(losses_G)], min(losses_G), 'o', color=axes_cycle[4], label='min G loss') 
    plt.xlabel("iteration") 
    plt.ylabel("loss") 
    plt.title("D loss vs G loss") 
    plt.legend() 
    plt.grid(True) 

    if show:
        plt.show()
    plt.savefig('result.png')

def show_graph(logfile, show):
    draw_results(*parse_log(logfile), show)

if __name__ == '__main__':
    LOG_FILE = 'log.txt'
    epoch, loss_d, loss_g = parse_log(LOG_FILE)
    draw_results(epoch, loss_d, loss_g)
