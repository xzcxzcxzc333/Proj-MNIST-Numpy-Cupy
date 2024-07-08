#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "include/mnist_file.h"
#include "include/neural_network.h"

#define STEPS 1000
#define BATCH_SIZE 100
#define TOTAL_EPOCHS 100  //增加的量
int STEP_OUT=0;
/**
 * Downloaded from: http://yann.lecun.com/exdb/mnist/
 */
const char * train_images_file = "data/train-images-idx3-ubyte";
const char * train_labels_file = "data/train-labels-idx1-ubyte";
const char * test_images_file = "data/t10k-images-idx3-ubyte";
const char * test_labels_file = "data/t10k-labels-idx1-ubyte";

/**
 * Calculate the accuracy of the predictions of a neural network on a dataset.
 */
float calculate_accuracy(mnist_dataset_t * dataset, neural_network_t * network)
{
    float activations[MNIST_LABELS], max_activation;
    int i, j, correct, predict;

    // Loop through the dataset
    for (i = 0, correct = 0; i < dataset->size; i++) {
        // Calculate the activations for each image using the neural network
        neural_network_hypothesis(&dataset->images[i], network, activations);

        // Set predict to the index of the greatest activation
        for (j = 0, predict = 0, max_activation = activations[0]; j < MNIST_LABELS; j++) {
            if (max_activation < activations[j]) {
                max_activation = activations[j];
                predict = j;
            }
        }

        // Increment the correct count if we predicted the right label
        if (predict == dataset->labels[i]) {
            correct++;
        }
    }

    // Return the percentage we predicted correctly as the accuracy
    return ((float) correct) / ((float) dataset->size);
}
//新加的函数
void log_metrics(float accuracy, float loss) {
    STEP_OUT++;
    FILE *fp = fopen("training_log.csv", "a");
    if (fp == NULL) {
        perror("Unable to open the file");
        return;
    }
    fprintf(fp, "%d,%.3f,%.3f\n", STEP_OUT, accuracy, loss);
    fclose(fp);
}
void print_progress(int step, int total_steps, int batch_index, int total_batches, float loss, float accuracy, time_t start_time) {
    int width = 50; // Total width for the progress bar
    float progress = (float)step / (float)total_steps;
    int pos = width * progress;
    int mins = (time(NULL) - start_time) / 60;
    int secs = (time(NULL) - start_time) % 60;
    float speed = (float)step / (float)(time(NULL) - start_time);
    int eta_min = (total_steps - step) / speed / 60;
    int eta_sec = (total_steps - step) / speed - eta_min * 60;

    printf("\rEpoch %d/%d: %3d%%|", (step / total_batches) + 1, TOTAL_EPOCHS, (int)(progress * 100));
    for (int i = 0; i < width; ++i) {
        if (i < pos) printf("█");
        else if (i == pos) printf("▋");
        else printf(" ");
    }
    if((int)(progress * 100)==0){
        printf("| %d/%d [%02d:%02d<??:??, ?.??it/s, Acc=%.3f, Loss=%.2f]", 
            batch_index + 1, total_batches, mins, secs , accuracy, loss / BATCH_SIZE);        
    }else{
        printf("| %d/%d [%02d:%02d<%02d:%02d, %.2fit/s, Acc=%.3f, Loss=%.2f]", 
            batch_index + 1, total_batches, mins, secs, eta_min, eta_sec, speed, accuracy, loss / BATCH_SIZE);
    }
    log_metrics(accuracy, loss / BATCH_SIZE);
    fflush(stdout);
}


int main(int argc, char *argv[])
{
    mnist_dataset_t * train_dataset, * test_dataset;
    mnist_dataset_t batch;
    neural_network_t network;
    float loss, accuracy;
    int i, batches;
    time_t start_time;
    // Read the datasets from the files
    train_dataset = mnist_get_dataset(train_images_file, train_labels_file);
    test_dataset = mnist_get_dataset(test_images_file, test_labels_file);

    // Initialise weights and biases with random values
    neural_network_random_weights(&network);

    // Calculate how many batches (so we know when to wrap around)
    batches = train_dataset->size / BATCH_SIZE;
    start_time = time(NULL);  //新加的
    for (i = 0; i < STEPS; i++) {
        // Initialise a new batch
        mnist_batch(train_dataset, &batch, 100, i % batches);

        // Run one step of gradient descent and calculate the loss
        loss = neural_network_training_step(&batch, &network, 0.5);

        // Calculate the accuracy using the whole test dataset
        accuracy = calculate_accuracy(test_dataset, &network);

        // printf("Step %04d\tAverage Loss: %.2f\tAccuracy: %.3f\n", i, loss / batch.size, accuracy);
        print_progress(i, STEPS, i % batches, batches, loss, accuracy, start_time);  //修改的
    }
    printf("\n");  // New line after progress finishes
    // Cleanup
    mnist_free_dataset(train_dataset);
    mnist_free_dataset(test_dataset);

    return 0;
}
