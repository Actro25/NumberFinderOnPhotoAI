#pragma once
void start_learning_machine(mnist::MNIST_dataset<uint8_t, uint8_t> data);

class AiNumberMachine
{
	//Вхід 784 нейрона по кожному на піксель картинки
	std::vector<float> InputLayer;
	//Заховані шари для обробки
	std::vector<std::vector<float>> HidenLayer;
	//Створення біасов для зміщення підчас розрахунку нових нейронів
	std::vector<std::vector<float>> HidenLayerBiases;
	//Ваги для обробми між шарами
	//[номер слоя][номер попереднього нейрона][номер наступного нейрона]
	std::vector<std::vector<std::vector<float>>> WeightForNeirons;
	//Виходячі нейрони. Їх буде 10
	//Кожен нейрон відповідає своїй цифрі від 0 - 9
	std::vector<float> OutputLayer;
	//Створення біасов для розрахунку кінцевих нейронів з зміщенням
	std::vector<float> OutputLayerBiases;
public:
	AiNumberMachine(int inputSize, int hidenLayerQuantity, 
		int quantityOfNeironsinHidenLayer) {
		//Записуємо розмір вхідного шару
		InputLayer.resize(inputSize);
		//Записуємо розмірм спрятаних шарів
		//Перше число це кількість шарів
		//Друга властивість це кількість нейронів типу float
		HidenLayer.resize(hidenLayerQuantity, std::vector<float>(quantityOfNeironsinHidenLayer));
		//Записуємо розміри для скритих шарів біасов
		HidenLayerBiases.resize(hidenLayerQuantity, std::vector<float>(quantityOfNeironsinHidenLayer));
		//Записуємо вихідний шар числом 10, тому що на виході повинно буди одне із 10 чисел
		OutputLayer.resize(10);
		//Записуємо розміри для виходячого шару біасов
		OutputLayerBiases.resize(10);
		//Перезаписуємо довжину ваг. Це записано кількість ваг які відповідає
		//кількості прихованих шарів + 1, тому що у нас є вхідні данні які підключаються
		//вагами до першого скеритого шару, і так само далі між кожним шаром нейронів
		// і тому виходить що кількість шарів ваг на 1 більше наж спратаних шарів.
		WeightForNeirons.resize(hidenLayerQuantity+1);
		
		//Цикл для запису довжин в вектор ваги в який ми записали кількість наших шарів ваг
		for (int i = 0; i < hidenLayerQuantity + 1;i++) {
			//Створюємо rows - це вхідні нейрони, cols - це вихідні нейрони.
			//Це робиться так тому що наші ваги пов'язані з нейронами. Один нейрон пов'язаний
			//з багатьома нейронами наступного шару нейронів 
			int rows, cols;
			//Перевіряємо якщо це перший входячи шар ваг то ми повинні записувати що перший шар ваг
			//буде пов'язаний між кількістю вхідних нейронів з кількістю нейронів першого скритого шару
			if (i == 0) {
				rows = inputSize;
				cols = quantityOfNeironsinHidenLayer;
			}
			//Перевіряємо якщо це останій шар ваг. Якщо це він то ми записуємо останій вектор ваг як
			//зв'язок між кількістю осанього скритого шару нейронів з вихідним шаром нейронів.
			else if (i == hidenLayerQuantity) {
				rows = quantityOfNeironsinHidenLayer;
				cols = 10;
			}
			//Записуємо базовий розмір
			else {
				rows = quantityOfNeironsinHidenLayer;
				cols = quantityOfNeironsinHidenLayer;
			}
			//Заповнюємо наш вектор ваг
			WeightForNeirons[i].resize(rows, std::vector<float>(cols));
		}
		//Налаштовуємо рандомайзер
		//Просимся до процесів комп'ютера щоб отримати випадкове число
		std::random_device rd;
		//Вмкористовуємо отримане число для запуска Вихр Марсенна для отримання потіка чисел
		std::mt19937 gen(rd());
		//Випадково записуємо числа від -0.5 до 0.5. Напотрібні маленькі числа щоб машина обучалася швидше
		std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

		//Робимо цикл для заповнення рандомними вагами числа щоб нейрона мережа сама обучалася та
		//покращувалася
		//Перша обгортка - це цикил який проходиться по кількості нашиг шарів ваг
		for (int i = 0;i < WeightForNeirons.size(); i++) {
			//Друга обгортка - це цикл який проходиться по першим нейронам які починаються зв'язок
			//з іншим нейроном
			for (int j = 0; j < WeightForNeirons[i].size(); j++) {
				//Третя обгортка - це цикл який роходиться по тим самим іншим нейронам які закривають 
				//зв'язок
				for (int q = 0; q < WeightForNeirons[i][j].size(); q++) {
					//Записуємо випадкове число
					WeightForNeirons[i][j][q] = dist(gen);
				}
			}
		}

	}
	int get_result() {
		int maxIndex = 0;
		float maxValue = OutputLayer[0];
		for (int i = 1; i < OutputLayer.size(); i++) {
			if (OutputLayer[i] > maxValue) {
				maxValue = OutputLayer[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	float funActivation(float sum) {
		//Якщо значення суми більше нуля то просто повертаємо sum інакше 0
		//Прочитайте більше про функцію ReLU як вона працює в інтернеті
		return (sum > 0) ? sum : 0;
	}
	void forward_propagation(int expectation_result) {
		//Починаємо обучення з обичного обрахунку нейронів
		//Створюємо прохід по скритим шарам
		for (int i = 0; i < HidenLayer.size(); i++) {
			//Перевіряємо чи це перший вхід бо якщо так ми харуємо нові нейрон задопомогою віхідного 
			//шару який ми передали
			if (i == 0) {
				//Спочатку проходемо по нейронам скритого шару,
				//бо на потрібно записати їм значення
				for (int j = 0; j < HidenLayer[i].size(); j++) {
					//Створюємо суму. Сума для кожного нейрона буде обраховуватися заново
					float sum = 0;
					//Проходимо по входячому вектору значеннь.
					for (int q = 0; q < InputLayer.size(); q++) {
						//Витягуємо значення ваги із вектора ваг. 
						//[i - шар ваги][q - входячий нейрон][j - відповідний нейрон наступного шару]
						//Сума рахується за формулою:
						//sum(aL) = funAct((aL-1*wi + bias))
						sum += InputLayer[q] * WeightForNeirons[i][q][j];
					}
					//Додаємо біас до суми для того щоб нейрон всерівно міг активуватися
					//навіть якщо прийдуть слабкі дані
					sum += HidenLayerBiases[i][j];
					//Рахуємо задопомогою функції активації (внашому випадку ReLU) суму
					HidenLayer[i][j] = funActivation(sum);
				}
			}
			//Всі інші варіанти коли проходимося по спрятаним шарам
			else{
				//Все робимо як у варіанті де i = 0, але
				//замість входячого шару нейронів у нас буде минулий шар скритих нейронів
				for (int j = 0; j < HidenLayer[i].size(); j++) {
					float sum = 0;
					for (int q = 0; q < HidenLayer[i - 1].size(); q++) {
						sum += HidenLayer[i - 1][q] * WeightForNeirons[i][q][j];
					}
					sum += HidenLayerBiases[i][j];
					HidenLayer[i][j] = funActivation(sum);
				}
			}
		}
		//Тут ми рахуємо кінцеві нейрони (виходячі)
		//Все майже схоже з минулими 2 варіантами але тут ми рахуємо між останім скритим шаром нейронів
		//і з виходним шаром нейронів
		for (int i = 0; i < OutputLayer.size(); i++) {
			float sum = 0;
			for (int j = 0; j < HidenLayer.back().size(); j++) {
				sum += HidenLayer.back()[j] * WeightForNeirons.back()[j][i];
			}
			sum += OutputLayerBiases[i];
			OutputLayer[i] = funActivation(sum);
		}
		//Отримаємо результат обробки
		float get_result();
	}
	void back_propagation() {}
};