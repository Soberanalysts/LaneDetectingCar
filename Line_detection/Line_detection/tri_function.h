float COS(int value, int distinguish)
{
	int COS[146] = { 0 };
	int i;

	COS[35] = 0.819;


	COS[40] = 0.766;


	COS[45] = 0.707;


	COS[50] = 0.643;


	COS[55] = 0.574;


	COS[60] = 0.500;


	COS[65] = 0.423;


	COS[70] = 0.342;


	COS[110] = 0.342;


	COS[115] = 0.423;


	COS[120] = 0.500;


	COS[125] = 0.574;


	COS[130] = 0.643;


	COS[135] = 0.707;


	COS[140] = 0.766;


	COS[145] = 0.819;

	return COS[value];
}
float SIN(int value,int distinguish)
{
	int SIN[146] = { 0 };
	int i;


	SIN[35] = 0.574;


	SIN[40] = 0.642;


	SIN[45] = 0.707;


	SIN[50] = 0.766;


	SIN[55] = 0.819;


	SIN[60] = 0.866;


	SIN[65] = 0.906;


	SIN[70] = 0.940;


	SIN[110] = 0.940;


	SIN[115] = 0.906;


	SIN[120] = 0.866;


	SIN[125] = 0.819;


	SIN[130] = 0.766;


	SIN[135] = 0.707;


	SIN[140] = 0.642;


	SIN[145] = 0.574;

	return SIN[value];
}