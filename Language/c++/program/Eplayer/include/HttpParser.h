//
// Created by buntu on 2023/8/13.
//

#ifndef EPLAYER_HTTPPARSER_H
#define EPLAYER_HTTPPARSER_H

#include "Socket.h"
#include "http_parser.h"
#include <map>

class CHttpParser
{
private:
    http_parser m_parser;
    http_parser_settings m_settings;
    std::map<Buffer, Buffer> m_HeaderValues;
    Buffer m_status;
    Buffer m_url;
    Buffer m_body;
    bool m_complete;
    Buffer m_lastField;
public:
    CHttpParser();
    ~CHttpParser();
    CHttpParser(const CHttpParser& http);       // 拷贝构造
    CHttpParser& operator=(const CHttpParser& http);    //赋值重载
public:
    size_t Parser(const Buffer& data);
    //GET POST ，参考http_parser.h HTTP_METHOD_MAP宏
    unsigned Method() const { return m_parser.method; }
    const std::map<Buffer, Buffer>& Headers() { return m_HeaderValues; }
    const Buffer& Status() const { return m_status; }
    const Buffer& Url() const { return m_url; }
    const Buffer& Body() const { return m_body; }
    unsigned Errno() const { return m_parser.http_errno; }
protected:
    // 这里的静态成员函数都有一个对应的成员函数相对应，便于成员函数中可以直接使用this
    static int OnMessageBegin(http_parser* parser);
    static int OnUrl(http_parser* parser, const char* at, size_t length);
    static int OnStatus(http_parser* parser, const char* at, size_t length);
    static int OnHeaderField(http_parser* parser, const char* at, size_t length);
    static int OnHeaderValue(http_parser* parser, const char* at, size_t length);
    static int OnHeadersComplete(http_parser* parser);
    static int OnBody(http_parser* parser, const char* at, size_t length);
    static int OnMessageComplete(http_parser* parser);
    int OnMessageBegin();
    int OnUrl(const char* at, size_t length);
    int OnStatus(const char* at, size_t length);
    int OnHeaderField(const char* at, size_t length);
    int OnHeaderValue(const char* at, size_t length);
    int OnHeadersComplete();
    int OnBody(const char* at, size_t length);
    int OnMessageComplete();
};

class UrlParser
{
public:
    UrlParser(const Buffer& url);
    ~UrlParser() {}
    int Parser();
    Buffer operator[](const Buffer& name)const;
    Buffer Protocol()const { return m_protocol; }
    Buffer Host()const { return m_host; }
    //Ä¬ÈÏ·µ»Ø80
    int Port()const { return m_port; }
    void SetUrl(const Buffer& url);
private:
    Buffer m_url;
    Buffer m_protocol;
    Buffer m_host;
    Buffer m_uri;
    int m_port;
    std::map<Buffer, Buffer> m_values;
};
#endif //EPLAYER_HTTPPARSER_H
