//
// Created by buntu on 2023/8/18.
//

#ifndef EPLAYER_CRYPTO_H
#define EPLAYER_CRYPTO_H
#include "Public.h"
#include <openssl/md5.h>
class Crypto
{
public:
    static Buffer MD5(const Buffer& text);
};

Buffer Crypto::MD5(const Buffer& text)
{
    Buffer result;
    Buffer data(16);
    MD5_CTX md5;
    MD5_Init(&md5);
    MD5_Update(&md5, text, text.size());
    MD5_Final(data, &md5);
    char temp[3] = "";
    for(size_t i=0;i<data.size();i++)
    {
        snprintf(temp, sizeof(temp), "%02x", data[i] & 0xFF);   // %02x，x小写才能转换为小写字母
        result += temp;
    }
    return result;
}

#endif //EPLAYER_CRYPTO_H
