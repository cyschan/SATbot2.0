from matplotlib import pyplot as plt

def plot_protocol_counts():
    x_points = [i for i in range(1, 21)]
    y_points = [2, 2, 19, 2, 1, 2, 8, 2, 2, 12, 2, 5, 2, 2, 3, 2, 0, 8, 0, 1]

    plt.xticks(range(1, len(x_points) + 1), x_points)
    plt.yticks(range(1, 20), range(1, 20))
    plt.xlabel('Protocols')
    plt.ylabel('Counts')
    plt.title('Protocol Counts')
    plt.bar(x_points, y_points, width=0.5)
    plt.show()

def plot_protocol_counts_feedback():
    width = 0.3
    better_x_points = [i for i in range(1, 21)]
    better_y_points = [2, 2, 14, 0, 1, 1, 6, 2, 1, 10, 4, 5, 1, 2, 1, 1, 1, 6, 1, 0]
    worse_x_points = [3 + width, 6 + width]
    worse_y_points = [2, 1]
    no_change_x_points = [i + 2 * width for i in range(1, 21)]
    no_change_y_points = [1, 0, 2, 2, 0, 0, 3, 0, 1, 1, 0, 0, 2, 0, 2, 1, 0, 1, 0, 1]
    plt.xticks(range(1, len(better_x_points) + 1), better_x_points)
    plt.yticks(range(1, 20), range(1, 20))
    plt.xlabel('Protocols')
    plt.ylabel('Counts')
    plt.title('Protocol Counts - How Users Felt After Attempting')
    
    plt.bar(better_x_points, better_y_points, width=0.3, color="green", label="Better")
    plt.bar(worse_x_points, worse_y_points, width=0.3, color="red", label="Worse")
    plt.bar(no_change_x_points, no_change_y_points, width=0.3, color="blue", label="No Change")
    plt.legend()
    plt.show()

# def plot_protocols_made_worse():
    
#     plt.xticks(range(0, 2), x_points)
#     plt.yticks(range(1, 3), range(1, 3))
#     plt.xlabel('Protocols')
#     plt.ylabel('Counts')
#     plt.title('Protocol Counts - User Felt Worse After Attempting')
#     plt.bar(range(len(y_points)), y_points, width=0.8)
#     plt.show()

# def plot_protocols_made_no_change():
    
#     plt.xticks(range(1, len(x_points) + 1), x_points)
#     plt.yticks(range(1, 5), range(1, 5))
#     plt.xlabel('Protocols')
#     plt.ylabel('Counts')
#     plt.title('Protocol Counts - No Change In Emotion After Attempting')
#     plt.bar(x_points, y_points, width=0.5)
#     plt.show()

# plot_protocol_counts()
plot_protocol_counts_feedback()
# plot_protocols_made_better()
# plot_protocols_made_worse()
# plot_protocols_made_no_change()