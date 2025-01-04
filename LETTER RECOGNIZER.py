import tkinter as tk
import numpy as np
import os

class LetterRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Letter Recognition GUI")

        # Parameters
        self.grid_size = (9, 7)  # Grid size for the input pattern
        self.input_size = self.grid_size[0] * self.grid_size[1]  # Total input size
        self.letters = ["A", "B", "C", "D", "E", "J", "K"]
        self.targets = {
            "A": [1, -1, -1, -1, -1, -1, -1],
            "B": [-1, 1, -1, -1, -1, -1, -1],
            "C": [-1, -1, 1, -1, -1, -1, -1],
            "D": [-1, -1, -1, 1, -1, -1, -1],
            "E": [-1, -1, -1, -1, 1, -1, -1],
            "J": [-1, -1, -1, -1, -1, 1, -1],
            "K": [-1, -1, -1, -1, -1, -1, 1],
        }
        self.threshold = 0.2  # Threshold value for activation
        self.learning_rate = 0.5
        self.max_iterations = 100  # Maximum iterations for training
        self.weights = {letter: np.zeros(self.input_size) for letter in self.letters}
        self.bias = {letter: 0 for letter in self.letters}

        # Load weights if available
        self.load_weights_from_txt()

        # Create widgets for GUI
        self.create_widgets()

    def create_widgets(self):
        # Grid Frame for Input Pattern
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.grid(row=0, column=0, padx=10, pady=10)

        self.grid_buttons = []
        for i in range(self.grid_size[0]):
            row_buttons = []
            for j in range(self.grid_size[1]):
                btn = tk.Button(self.grid_frame, width=2, height=1, bg="white", 
                                command=lambda i=i, j=j: self.toggle_grid(i, j))
                btn.grid(row=i, column=j)
                row_buttons.append(btn)
            self.grid_buttons.append(row_buttons)

        # Right Side Controls
        self.control_frame = tk.Frame(self.root)
        self.control_frame.grid(row=0, column=1, padx=10, pady=10)

        self.target_label = tk.Label(self.control_frame, text="Target Letter:")
        self.target_label.grid(row=0, column=0)

        self.target_entry = tk.Entry(self.control_frame)
        self.target_entry.grid(row=0, column=1)

        self.learning_rate_label = tk.Label(self.control_frame, text="Learning Rate:")
        self.learning_rate_label.grid(row=1, column=0)
        
        self.learning_rate_entry = tk.Entry(self.control_frame)
        self.learning_rate_entry.insert(0, str(self.learning_rate))
        self.learning_rate_entry.grid(row=1, column=1)

        self.train_button = tk.Button(self.control_frame, text="Train", command=self.train)
        self.train_button.grid(row=2, column=0, columnspan=2)

        self.test_button = tk.Button(self.control_frame, text="Test", command=self.test)
        self.test_button.grid(row=3, column=0, columnspan=2)

        self.save_button = tk.Button(self.control_frame, text="Save Weights", command=self.save_weights_to_txt)
        self.save_button.grid(row=4, column=0, columnspan=2)

        # Output Display
        self.output_frame = tk.Frame(self.control_frame, borderwidth=1, relief="solid")
        self.output_frame.grid(row=5, column=0, columnspan=2, pady=10)

        self.output_label = tk.Label(self.output_frame, text="Output:")
        self.output_label.grid(row=0, column=0)

        self.output_text = tk.Label(self.output_frame, text="", width=20)
        self.output_text.grid(row=0, column=1)

        self.result_label = tk.Label(self.control_frame, text="Prediction:")
        self.result_label.grid(row=6, column=0)

        self.result_value = tk.Label(self.control_frame, text="")
        self.result_value.grid(row=6, column=1)

        # Button to clear grid input
        self.clear_grid_button = tk.Button(self.control_frame, text="Clear Grid", command=self.clear_grid)
        self.clear_grid_button.grid(row=7, column=0, columnspan=2)

        # Button to reset training count
        self.reset_training_button = tk.Button(self.control_frame, text="Reset Training", command=self.reset_training)
        self.reset_training_button.grid(row=8, column=0, columnspan=2)

    def toggle_grid(self, i, j):
        btn = self.grid_buttons[i][j]
        current_color = btn['bg']
        btn['bg'] = 'blue' if current_color == 'white' else 'white'

    def get_input_vector(self):
        input_vector = np.full((self.grid_size[0], self.grid_size[1]), -1)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.grid_buttons[i][j]['bg'] == 'blue':
                    input_vector[i, j] = 1
        return input_vector.flatten()

    def train(self):
        input_vector = self.get_input_vector()
        input_data = {letter: [input_vector] for letter in self.letters}
        target_letter = self.target_entry.get().upper()
        
        if target_letter in self.letters:
            self.learning_rate = float(self.learning_rate_entry.get())
            trained, epochs = self.perceptron_train(input_data, target_letter)

            # Output training results
            if trained:
                self.result_value.config(text=f"Trained on letter {target_letter}")
            else:
                self.result_value.config(text=f"Training Complete in {epochs} epochs")
            
            # Display target vector in output
            target_vector = self.targets[target_letter]
            self.output_text.config(text=" ".join(map(str, target_vector)))
        else:
            self.result_value.config(text="Invalid Target Letter")

    def perceptron_train(self, input_data, target_letter):
        epoch = 0
        trained = False
        target = self.targets[target_letter]

        while not trained and epoch < self.max_iterations:
            trained = True
            for letter, sample_list in input_data.items():
                for sample in sample_list:
                    y_in = self.bias[letter]
                    for i in range(self.input_size):
                        y_in += sample[i] * self.weights[letter][i]

                    # Activation function updated to include 0
                    if y_in > self.threshold:
                        y = 1
                    elif y_in < self.threshold:
                        y = -1
                    else:
                        y = 0

                    if y != target[self.letters.index(letter)]:
                        trained = False
                        error = target[self.letters.index(letter)] - y
                        self.bias[letter] += self.learning_rate * error
                        for i in range(self.input_size):
                            self.weights[letter][i] += self.learning_rate * sample[i] * error
            epoch += 1

        return trained, epoch

    def test(self):
        input_vector = self.get_input_vector()
        predictions = self.perceptron_test(input_vector)

        # Display predicted letters and corresponding target vector
        if predictions:
            predicted_letter = predictions[0]
            target_vector = self.targets[predicted_letter]
            self.result_value.config(text=predicted_letter)
            self.output_text.config(text=" ".join(map(str, target_vector)))
        else:
            self.result_value.config(text="No Match")
            self.output_text.config(text="")

    def perceptron_test(self, input_vector):
        found = []
        for letter, weight in self.weights.items():
            y_in = self.bias[letter]
            for i in range(self.input_size):
                y_in += input_vector[i] * weight[i]
            if y_in > self.threshold:
                found.append(letter)
        return found

    def save_weights_to_txt(self):
        with open("weights.txt", "w") as f:
            for letter, weight in self.weights.items():
                f.write(f"Letter: {letter}\n")
                f.write("Weights: " + " ".join(map(str, weight)) + "\n")
                f.write(f"Bias: {self.bias[letter]}\n")
                f.write("\n")
        print("Weights saved to weights.txt!")

    def load_weights_from_txt(self):
        if not os.path.exists("weights.txt"):
            print("No saved weights file found.")
            return

        with open("weights.txt", "r") as f:
            lines = f.readlines()
            current_letter = None
            for line in lines:
                if line.startswith("Letter:"):
                    current_letter = line.split(":")[1].strip()
                    self.weights[current_letter] = []
                elif line.startswith("Weights:"):
                    weights = list(map(float, line.split(":")[1].strip().split()))
                    self.weights[current_letter] = np.array(weights)
                elif line.startswith("Bias:"):
                    self.bias[current_letter] = float(line.split(":")[1].strip())

    def clear_grid(self):
        for row in self.grid_buttons:
            for btn in row:
                btn.config(bg="white")

    def reset_training(self):
        self.weights = {letter: np.zeros(self.input_size) for letter in self.letters}
        self.bias = {letter: 0 for letter in self.letters}
        self.result_value.config(text="Training Reset")

# Main program
if __name__ == "__main__":
    root = tk.Tk()
    app = LetterRecognitionApp(root)
    root.mainloop()