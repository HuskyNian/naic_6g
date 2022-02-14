# naic_6g <br>
2022naic 6g competition


environment:<br>
•	numpy==1.19.5<br>
•	torch=1.9.1<br>
<br>
Training model:<br>
1. 将数据放在同一目录下，结构为:<br>
/train:<br>
--/Htrain.mat<br>
--/Htest.mat<br>
<br>
2.	run Model_train.py<br>
3.	在models文件夹中含有生成的结果encoder.pth和decoder.pth<br>
4.	然后把ModelDefine.py，encoder.pth和decoder.pth放到project文件夹里，打包提交即可。

