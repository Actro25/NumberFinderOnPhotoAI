#pragma once
std::vector<float> load_image_as_pixels(std::string path) {
    int width, height, channels;
    unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 1);

    if (img == NULL) {
        std::cout << "Error loading image!" << std::endl;
        return {};
    }

    // Перевіряємо, чи підходить розмір
    if (width != 28 || height != 28) {
        std::cout << "Warning: Image is " << width << "x" << height << ". Needs to be 28x28!" << std::endl;
    }

    std::vector<float> pixels;
    for (int i = 0; i < width * height; i++) {
        pixels.push_back(1.0f - (img[i] / 255.0f));
    }

    stbi_image_free(img);
    return pixels;
}
void calculate_accuracy(AiNumberMachine& net, const mnist::MNIST_dataset<uint8_t, uint8_t>& data) {
    int correct_predictions = 0;
    int total_images = data.test_images.size();

    std::cout << "Checking accuracy on " << total_images << " test images..." << std::endl;

    for (int i = 0; i < total_images; i++) {
        std::vector<float> input(784);
        for (int j = 0; j < 784; j++) {
            input[j] = data.test_images[i][j] / 255.0f;
        }

        net.predict(input);
        int ai_answer = net.get_result();
        int real_answer = (int)data.test_labels[i];

        if (ai_answer == real_answer) {
            correct_predictions++;
        }
    }

    float accuracy = (float)correct_predictions / total_images * 100.0f;
    std::cout << "------------------------------------" << std::endl;
    std::cout << "Correct: " << correct_predictions << " out of " << total_images << std::endl;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    std::cout << "------------------------------------" << std::endl;
}

void start_learning_machine(mnist::MNIST_dataset<uint8_t, uint8_t> data) {
	AiNumberMachine ourAi(784, 2, {64,32}); //AiNumberMachine ourAi(3, 1, { 2 });
	std::vector<std::vector<float>> normalized_data(data.training_images.size(),std::vector<float>(784));
	std::vector<float> prepared_expectation_data(10);
	for (int i = 0; i < data.training_images.size(); i++) {
		for (int j = 0; j < data.training_images[i].size(); j++) {
			normalized_data[i][j] = (data.training_images[i][j] / 255.0f);
		}
	}
	
	for (int i = 0; i < normalized_data.size(); i++) {

		std::vector<float> target(10, 0.0f);
		int correctDigit = (int)data.training_labels[i];
		target[correctDigit] = 1.0f;

		ourAi.forward_propagation(normalized_data[i], target);

		if (i % 1000 == 0) std::cout << "Processed: " << i << " images" << std::endl;
	}

    calculate_accuracy(ourAi, data);

    std::vector<float> my_digit = load_image_as_pixels("three.png");


    ourAi.forward_propagation(my_digit, std::vector<float>(10, 0.0f));


    int result = ourAi.get_result();
    std::cout << "I think this is number: " << result << std::endl;

}