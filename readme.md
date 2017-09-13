Economy to Lang
====

## Overview
Economy to langは株価データから経済記事を出力するのを目標とする。
経済を題材に、単文ではなく複数の文を生成するのが最大の課題となる。

## 短期的目標
 [太田]言語モデリングのサーベイ。RNNGの実装をめざす。
 段階ごとに分けられ、
 1. transition based parsing[arc-eager]
 2. transition based parsing[stacked LSTM]
 3. Reccurent Neural Network Grammer  
の順に行う考え。1番目は2番目に、2番目は3番目に強い内容的連環をもっている。
この話題はparsing/以下で取り扱っている。現在は1を達成し2に取りかかるところ。

## Resource
 auto以下に整形されたデータを出力するスクリプトです。 
 - bin/OHLC.py 
 - bin/extract.py 
 
 auto以下は次のような構成になっています。
  - auto/dj39/*.tsv  
  - auto/djnml_daily_headline/*.txt 

  tsvファイルは、その日の、05-14年,１社あたりの[open,high,low,close]の値を示します(これをOHLCや四本足と呼びます）。 
  txtファイルは、その日の、ダウ・ジョーンズ通信によるヘッドライン記事です。１行目は見出しを示します。 

 resource/以下はNIIによる元データが収まっています。 
  corpus/以下にはダウ・ジョーンズ通信の記事ファイルが、 
  numerical/にはダウ工業平均指数に期間中指定された銘柄の株価が収められています。 
  なお、numerical/以下のtsvは、それぞれ[企業名,日時,時刻,株価,約定,出来高]を示しています。 


## usage

## Install

## Licence
private

## Author
