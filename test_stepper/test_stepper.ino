int Ydirpin = 6;
int Ysteppin = 3;
int Xdirpin = 5;
int Xsteppin = 2;
int en = 8;
void setup()
{
  pinMode(Ydirpin,OUTPUT);
  pinMode(Ysteppin,OUTPUT);
  pinMode(Xdirpin,OUTPUT);
  pinMode(Xsteppin,OUTPUT);
  pinMode(en,OUTPUT);
  digitalWrite(en, LOW);
}
void loop()
{
  int j;
  delayMicroseconds(2);
  digitalWrite(Ydirpin,LOW);
  digitalWrite(Xdirpin,LOW);
  for(j=0;j<=5000;j  ){
    digitalWrite(Ysteppin,LOW);
    digitalWrite(Xsteppin,LOW);
    delayMicroseconds(2);
    digitalWrite(Ysteppin,HIGH);
    digitalWrite(Xsteppin,HIGH);
    delay(1);
  }
}