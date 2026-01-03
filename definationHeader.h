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
	
	float LearningRate = 0.005;
public:
	AiNumberMachine(int inputSize, int hidenLayerQuantity, 
		std::vector<int> quantityOfNeironsinHidenLayer) {
		//Записуємо розмір вхідного шару
		InputLayer.resize(inputSize);
		//Записуємо розмірм спрятаних шарів
		//Перше число це кількість шарів
		//Запвонюємо щари спрятаного шару
		HidenLayer.resize(hidenLayerQuantity);
		//Тепер записуємо ту кількість нейрон яку було передано для шарів через вектор
		for (int i = 0; i < HidenLayer.size(); i++) {
			HidenLayer[i].resize(quantityOfNeironsinHidenLayer[i]);
		}
		//Записуємо розміри для скритих шарів біасов
		HidenLayerBiases.resize(hidenLayerQuantity);
		//Робимо те саме що і з спрятаним шаром
		for (int i = 0; i < HidenLayerBiases.size(); i++) {
			HidenLayerBiases[i].resize(quantityOfNeironsinHidenLayer[i]);
		}
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
				cols = HidenLayer[0].size();
			}
			//Перевіряємо якщо це останій шар ваг. Якщо це він то ми записуємо останій вектор ваг як
			//зв'язок між кількістю останього скритого шару нейронів з вихідним шаром нейронів.
			else if (i == hidenLayerQuantity) {
				rows = HidenLayer[i - 1].size();
				cols = 10;
			}
			//Записуємо всі інші випадки
			else {
				rows = HidenLayer[i - 1].size();
				cols = HidenLayer[i].size();
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
		std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

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
		//Заповнюємо біас для вихідного шару випадковими числами
		for (int i = 0; i < OutputLayerBiases.size(); i++) {
			OutputLayerBiases[i] = dist(gen);
		}
		//Тепер заповнюємо біаси для скритих слоїв
		for (int i = 0; i < HidenLayerBiases.size(); i++) {
			for (int j = 0; j < HidenLayerBiases[i].size(); j++) {
				HidenLayerBiases[i][j] = dist(gen);
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
	float derivativeReLU(float x) {
		//Повертаємо похідну функції
		return x > 0 ? 1.0f : 0.0f;
	}
	void back_propagation(std::vector<float> Target) {
		//Створюємо вектор для помилок виходних нейронів
		std::vector<float> OutputDeltas(OutputLayer.size());
		//Створюємо двовимірний вектор для запису помилок нейронів у шарах
		std::vector<std::vector<float>> HidenDeltas(HidenLayer.size());
		for (int i = 0; i < HidenLayer.size(); i++) {
			HidenDeltas[i].resize(HidenLayer[i].size());
		}
		//Для почвтку ми знайдемо наші помилки для вихідного шару за формулою:
		//Помилка для ваг та нейронів = похідна помилки помножена на похідну функції
		for (int i = 0; i < OutputLayer.size(); i++) {
			OutputDeltas[i] = (OutputLayer[i] - Target[i]); //ТУТ БУЛО ПРИБРАНО ПОХІДНУ ФУНКЦІЇ
		}
		//Записуємо основний цикл по якому будемо ідти
		//Так як ми повертаємося назад то і відлік буде йти назад
		for (int k = WeightForNeirons.size() - 1; k >= 0; k--) {
			//Перевіряємо чи на шар часом не пов'язаний з вихідним шаром
			if (k == WeightForNeirons.size() - 1) {
				//Тут ми рахуємо помилку для наступного кроку щоб було легше
				for (int i = 0; i < HidenLayer.back().size(); i++) {
					//Створюємо суму помилок, тому що однин нейрон спрятаного шару пов'язаний з усіма нейронами іншого гару.
					//І тому ми рахуємо всі помилки які він завдав і іншим нейронам
					float sumError = 0;
					//Проходимося по вихідному шару і шукаємо суму всі нейронів 
					//помножених на вагу яка їх з'єднує з спрятаним шаром
					for (int j = 0; j < OutputLayer.size(); j++) {
						sumError += OutputDeltas[j] * WeightForNeirons.back()[i][j];
					}
					//Тут ми убираємо вплив функції на помилку.
					//Бо нам потрібна чиста помилка і тому ми шукаємо похідну функції і множимо на суму помилок і записуємо її.
					HidenDeltas.back()[i] = sumError * derivativeReLU(HidenLayer.back()[i]);
				}
				//Тут ми змінюємо наші ваги на кращі які будуть робити менші помилки
				//Ми беремо в цьому циклі нейрони з точки де ми знаходимося
				for (int i = 0; i < WeightForNeirons.back().size(); i++)
				{
					//В цьому циклі ми беремо ті нейрони з якими ми пов'язані
					for (int f = 0; f < WeightForNeirons.back()[i].size(); f++) {
						//Рахуємо нові ваги за формулою:
						//Нова вага = стара вага - (швидкість вчіння * градієнт помилки)
						//Градієнт помилки розраховувається за формулою: Градієнт помилки = похідна помилки * похідну функції активації * вхідний нейрон
						//Так як ми вже розрахували одну части цієї формули вище то нам потрібно просто підставити вхідний нейрон (нейрон на якому ми зараз знаходимося)
						WeightForNeirons.back()[i][f] -= (LearningRate * OutputDeltas[f] * HidenLayer.back()[i]);
					}
				}
				//Оновлюємо біас просто за формулою: Новий біас = старий біас - (швидкість вчиння * помилку того нейрона до якого пренадлежить біас)
				for (int i = 0; i < OutputLayerBiases.size(); i++) {
					OutputLayerBiases[i] -= (LearningRate * OutputDeltas[i]);
				}
			}
			//Тут ми перевіряємо чи часом це не кінець, бо якщо кінець то ми пов'язані з вхідними данними
			else if (k == 0) {
				//Ми можемо не шукати помилки для наступних гарів томущо це вже кінцева зупинка
				//Тут ми шукаємо новів ваги але ми тепер працюємо з вхідними данними
				for (int i = 0; i < WeightForNeirons[0].size(); i++)
				{
					for (int f = 0; f < WeightForNeirons[0][i].size(); f++) {
						WeightForNeirons[0][i][f] -= (LearningRate * InputLayer[i] * HidenDeltas[0][f]);
					}
				}
				//Тут ми просто оновлюємо біас по тій самі формулі
				for (int i = 0; i < HidenLayerBiases[0].size(); i++) {
					HidenLayerBiases[0][i] -= (LearningRate * HidenDeltas[0][i]);
				}
			}
			//Всі інші випадки
			else {
				//Обраховуємо помилку для наступного шару по тій самі формулі що була вище
				for (int i = 0; i < HidenLayer[k - 1].size(); i++) {
					float sumError = 0;
					for (int j = 0; j < HidenLayer[k].size(); j++) {
						sumError += HidenDeltas[k][j] * WeightForNeirons[k][i][j];
					}
					HidenDeltas[k - 1][i] = sumError * derivativeReLU(HidenLayer[k - 1][i]);
				}
				//Так само оновлюємо ваги як і вище
				for (int i = 0; i < WeightForNeirons[k].size(); i++)
				{
					for (int f = 0; f < WeightForNeirons[k][i].size(); f++) {
						WeightForNeirons[k][i][f] -= (LearningRate * HidenLayer[k - 1][i] * HidenDeltas[k][f]);
					}
				}
				//Оновлюємо біас по тій самій формулі як і вище
				for (int i = 0; i < HidenLayerBiases[k].size(); i++) {
					HidenLayerBiases[k][i] -= (LearningRate * HidenDeltas[k][i]);
				}
			}
		}
	}
	void forward_propagation(std::vector<float> input_data, std::vector<float> expectation_data) {
		//Перезаписуємо вхідні данні на ті які ми передаємо для обучення
		InputLayer = input_data;
		//Починаємо обучення з обичного обрахунку нейронів
		//Створюємо прохід по скритим шарам
		for (int i = 0; i < HidenLayer.size(); i++) {
			//Перевіряємо чи це перший вхід бо якщо так ми рахуємо нові нейрон задопомогою віхідного 
			//шару який ми передали
			if (i == 0) {
				//Спочатку проходемо по нейронам скритого шару,
				//бо нам потрібно записати їм значення
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
			OutputLayer[i] = sum;
		}

		//Запускаємо обробку помилок які ми зробили та передаємо вектор очікуваної відповіді
		back_propagation(expectation_data);
	}
	void predict(const std::vector<float>& input_data) {
		InputLayer = input_data;
		for (int i = 0; i < HidenLayer.size(); i++) {
			if (i == 0) {
				for (int j = 0; j < HidenLayer[i].size(); j++) {
					float sum = 0;
					for (int q = 0; q < InputLayer.size(); q++) {
						sum += InputLayer[q] * WeightForNeirons[i][q][j];
					}
					sum += HidenLayerBiases[i][j];
					HidenLayer[i][j] = funActivation(sum);
				}
			}
			else {
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
		for (int i = 0; i < OutputLayer.size(); i++) {
			float sum = 0;
			for (int j = 0; j < HidenLayer.back().size(); j++) {
				sum += HidenLayer.back()[j] * WeightForNeirons.back()[j][i];
			}
			sum += OutputLayerBiases[i];
			OutputLayer[i] = sum;
		}

	}
};