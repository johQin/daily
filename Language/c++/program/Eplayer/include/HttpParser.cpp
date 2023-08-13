//
// Created by buntu on 2023/8/13.
//
#include "HttpParser.h"

CHttpParser::CHttpParser()
{
    m_complete = false;
    memset(&m_parser, 0, sizeof(m_parser));
    m_parser.data = this;
    http_parser_init(&m_parser, HTTP_REQUEST);      //http_parser_type { HTTP_REQUEST, HTTP_RESPONSE, HTTP_BOTH };
    memset(&m_settings, 0, sizeof(m_settings));

    // 对应于 http_parser_settings,用于设置各种回调，这里的回调使用的是静态成员函数，
    // 因为http_parser是c语言，是面向过程的语言，所以无法使用this，
    // 这里使用静态成员函数，而静态成员函数里面又通过this调用非静态成员函数，这样就修改为面向对象的了
    m_settings.on_message_begin = &CHttpParser::OnMessageBegin;
    m_settings.on_url = &CHttpParser::OnUrl;
    m_settings.on_status = &CHttpParser::OnStatus;
    m_settings.on_header_field = &CHttpParser::OnHeaderField;
    m_settings.on_header_value = &CHttpParser::OnHeaderValue;
    m_settings.on_headers_complete = &CHttpParser::OnHeadersComplete;
    m_settings.on_body = &CHttpParser::OnBody;
    m_settings.on_message_complete = &CHttpParser::OnMessageComplete;
}

CHttpParser::~CHttpParser()
{}

CHttpParser::CHttpParser(const CHttpParser& http)
{
    memcpy(&m_parser, &http.m_parser, sizeof(m_parser));
    m_parser.data = this;
    memcpy(&m_settings, &http.m_settings, sizeof(m_settings));
    m_status = http.m_status;
    m_url = http.m_url;
    m_body = http.m_body;
    m_complete = http.m_complete;
    m_lastField = http.m_lastField;
}

CHttpParser& CHttpParser::operator=(const CHttpParser& http)
{
    if (this != &http) {
        memcpy(&m_parser, &http.m_parser, sizeof(m_parser));
        m_parser.data = this;
        memcpy(&m_settings, &http.m_settings, sizeof(m_settings));
        m_status = http.m_status;
        m_url = http.m_url;
        m_body = http.m_body;
        m_complete = http.m_complete;
        m_lastField = http.m_lastField;
    }
    return *this;
}

size_t CHttpParser::Parser(const Buffer& data)
{
    m_complete = false;
    size_t ret = http_parser_execute(
            &m_parser, &m_settings, data, data.size());
    if (m_complete == false) {
        m_parser.http_errno = 0x7F;
        return 0;
    }
    return ret;
}
// 从类的静态成员函数转调用到成员函数，这样函数里面就可以直接用this
int CHttpParser::OnMessageBegin(http_parser* parser)
{
    return ((CHttpParser*)parser->data)->OnMessageBegin();
}

int CHttpParser::OnUrl(http_parser* parser, const char* at, size_t length)
{
    return ((CHttpParser*)parser->data)->OnUrl(at, length);
}

int CHttpParser::OnStatus(http_parser* parser, const char* at, size_t length)
{
    return ((CHttpParser*)parser->data)->OnStatus(at, length);
}

int CHttpParser::OnHeaderField(http_parser* parser, const char* at, size_t length)
{
    return ((CHttpParser*)parser->data)->OnHeaderField(at, length);
}

int CHttpParser::OnHeaderValue(http_parser* parser, const char* at, size_t length)
{
    return ((CHttpParser*)parser->data)->OnHeaderValue(at, length);
}

int CHttpParser::OnHeadersComplete(http_parser* parser)
{
    return ((CHttpParser*)parser->data)->OnHeadersComplete();
}

int CHttpParser::OnBody(http_parser* parser, const char* at, size_t length)
{
    return ((CHttpParser*)parser->data)->OnBody(at, length);
}

int CHttpParser::OnMessageComplete(http_parser* parser)
{
    return ((CHttpParser*)parser->data)->OnMessageComplete();
}

int CHttpParser::OnMessageBegin()
{
    return 0;
}

int CHttpParser::OnUrl(const char* at, size_t length)
{
    m_url = Buffer(at, length);
    return 0;
}

int CHttpParser::OnStatus(const char* at, size_t length)
{
    m_status = Buffer(at, length);
    return 0;
}

int CHttpParser::OnHeaderField(const char* at, size_t length)
{
    m_lastField = Buffer(at, length);
    return 0;
}

int CHttpParser::OnHeaderValue(const char* at, size_t length)
{
    m_HeaderValues[m_lastField] = Buffer(at, length);
    return 0;
}

int CHttpParser::OnHeadersComplete()
{
    return 0;
}

int CHttpParser::OnBody(const char* at, size_t length)
{
    m_body = Buffer(at, length);
    return 0;
}

int CHttpParser::OnMessageComplete()
{
    m_complete = true;
    return 0;
}

UrlParser::UrlParser(const Buffer& url)
{
    m_url = url;
}

int UrlParser::Parser()
{
    //分三步：协议、域名和端口、uri、键值对
    //解析协议
    const char* pos = m_url;
    const char* target = strstr(pos, "://");
    if (target == NULL)return -1;
    m_protocol = Buffer(pos, target);
    //解析域名和端口
    pos = target + 3;
    target = strchr(pos, '/');
    if (target == NULL) {
        if (m_protocol.size() + 3 >= m_url.size())
            return -2;
        m_host = pos;
        return 0;
    }
    Buffer value = Buffer(pos, target);
    if (value.size() == 0)return -3;
    target = strchr(value, ':');
    if (target != NULL) {
        m_host = Buffer(value, target);
        m_port = atoi(Buffer(target + 1, (char*)value + value.size()));
    }
    else {
        m_host = value;
    }
    pos = strchr(pos, '/');
    //解析uri
    target = strchr(pos, '?');
    if (target == NULL) {
        m_uri = pos;
        return 0;
    }
    else {
        m_uri = Buffer(pos, target);
        //解析key和value
        pos = target + 1;
        const char* t = NULL;
        do {
            target = strchr(pos, '&');
            if (target == NULL) {
                t = strchr(pos, '=');
                if (t == NULL)return -4;
                m_values[Buffer(pos, t)] = Buffer(t + 1);
            }
            else {
                Buffer kv(pos, target);
                t = strchr(kv, '=');
                if (t == NULL)return -5;
                m_values[Buffer(kv, t)] = Buffer(t + 1, kv + kv.size());
                pos = target + 1;
            }
        } while (target != NULL);
    }

    return 0;
}

Buffer UrlParser::operator[](const Buffer& name) const
{
    auto it = m_values.find(name);
    if (it == m_values.end())return Buffer();
    return it->second;
}

void UrlParser::SetUrl(const Buffer& url)
{
    m_url = url;
    m_protocol = "";
    m_host = "";
    m_port = 80;
    m_values.clear();
}
