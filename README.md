# COP3530-Project-3

Final project for COP3530. A Marketstack API key is required (signup offers 100 free requests).
This project uses tensorflow's linear regression to predict stock prices, then displays the predicted close price (using CustomTkinter) along with a graph (Marketstack API) of the actual closing price for user convenience.
This project also contains an unused custom LSTM implementation, which is made for educational purposes only.

Example usage:
![image](https://github.com/user-attachments/assets/75c3ce33-a961-4a2e-9577-e19093e313ac)

![image](https://github.com/user-attachments/assets/f4461acc-a90a-421d-9318-6c6898f70746)

![image](https://github.com/user-attachments/assets/2c5354f6-a069-46c0-af28-adc5b92cc50e)

![image](https://github.com/user-attachments/assets/8346ab7d-c096-413a-a044-810699f57c44)

![image](https://github.com/user-attachments/assets/f07b1971-937b-4228-8cec-dddbfc567737)

**A log of my progress** 

First version unoptimized: 
Loss ~ 45.77
Training time ~ 34h

Second version unoptimized: 
Loss ~ 12.21
Training time ~ 12h

Third version unoptimized: 
Loss ~ 2.31
Training time ~ 3h

Fourth version unoptimized
Loss ~ 1.39
Training time ~ 26m
![image](https://github.com/user-attachments/assets/f2f7cdb1-ba83-40c4-984b-38a54cb129c4)

First version (Custom LSTM) optimized (Ignore this crime on humanity)
Loss ~ 10^5
Training time ~ 4.58h
![image](https://github.com/user-attachments/assets/87c1c524-f950-48d3-bf3a-ee668e0b1121)

First version (keras LSTM) optimized
Loss ~ 1.38
Training time ~ 3m 56s
![image](https://github.com/user-attachments/assets/002d7099-a69b-4bab-abc4-2cc6e6c44559)

Second version (keras LSTM) optimized
Loss ~ 0.98
Training time ~ 20 m
![image](https://github.com/user-attachments/assets/b90b7856-31d4-4947-b92a-c8f448ac2789)


Demonstration (Input is a ticker): ![image](https://github.com/user-attachments/assets/d0b9c175-fc96-4bf7-8824-2182859dff71)

