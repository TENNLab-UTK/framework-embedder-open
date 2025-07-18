#include <string>

class IndentString {
  public:
    IndentString();
    IndentString(const std::string &s, unsigned int spaces = 0);
    IndentString &operator=(const std::string &s);
    IndentString &operator+=(const std::string &s);
    IndentString &operator+=(IndentString &is);
    IndentString operator+(const std::string &s);
    IndentString operator+(IndentString &is);

    std::string get_str();
    void set_indent_spaces(unsigned int spaces);
    void add_indent_spaces(int spaces);

    void append(const std::string &s);
    void append(IndentString &is);

  protected:
    std::string str;
    unsigned int indent_spaces;

    std::string indent_lines(const std::string &preceding_str,
                             const std::string &unindented_str);
};