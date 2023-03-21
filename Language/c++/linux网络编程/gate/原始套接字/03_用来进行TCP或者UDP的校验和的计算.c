//=========================================================================
//函数:unsigned short checksum(unsigned short *buf, int nword)
//作用:用来进行TCP或者UDP的校验和的计算
//参数:
//		buf:进行校验和的数据的起始地址
//		nword:进行校验的数据的个数(注意本函数是2个Byte进行校验，所以nword应该为
//			实际数据个数的一半)
// UDP检验和的计算方法是：
// 1.按每16位求和得出一个32位的数；
// 2.如果这个32位的数，高16位不为0，则高16位加低16位再得到一个32位的数；
// 3.重复第2步直到高16位为0，将低16位取反，得到校验和。
//=========================================================================
unsigned short checksum(unsigned short *buf, int nword)
{
	unsigned long sum;
	for(sum = 0; nword > 0; nword--)
	{
		sum += htons(*buf);
		buf++;
	}
	sum = (sum>>16) + (sum&0xffff);
	sum += (sum>>16);
	return ~sum;
}