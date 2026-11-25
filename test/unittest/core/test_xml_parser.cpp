#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <unistd.h>

#include <gtest/gtest.h>

#include "xml.h"

namespace {

flagcxResult_t parseXmlText(const std::string &contents, flagcxXml **xml) {
  *xml = nullptr;
  char path[] = "/tmp/flagcx-xml-XXXXXX";
  int fd = mkstemp(path);
  if (fd < 0)
    return flagcxSystemError;

  FILE *file = fdopen(fd, "w");
  if (file == nullptr) {
    close(fd);
    unlink(path);
    return flagcxSystemError;
  }

  const size_t written = fwrite(contents.data(), 1, contents.size(), file);
  const int closeResult = fclose(file);
  if (written != contents.size() || closeResult != 0) {
    unlink(path);
    return flagcxSystemError;
  }

  flagcxXml *parsed = nullptr;
  flagcxResult_t result = xmlAlloc(&parsed, 8);
  if (result == flagcxSuccess)
    result = flagcxTopoGetXmlFromFile(path, parsed, 0);
  unlink(path);

  if (result != flagcxSuccess) {
    free(parsed);
    return result;
  }
  *xml = parsed;
  return flagcxSuccess;
}

TEST(XmlParser, ParsesBothQuoteStylesAndXmlWhitespace) {
  const std::string contents =
      "<system\tversion \t= '1'\r\n name = \"FlagCX parser test\" >\r\n"
      "\t<gpu\tdev = '0'\trank=\"1\" note='contains \"double quotes\"' />\r\n"
      "</system>\r\n";

  flagcxXml *rawXml = nullptr;
  ASSERT_EQ(parseXmlText(contents, &rawXml), flagcxSuccess);
  std::unique_ptr<flagcxXml, decltype(&std::free)> xml(rawXml, &std::free);

  ASSERT_EQ(xml->maxIndex, 2);
  const flagcxXmlNode &system = xml->nodes[0];
  const flagcxXmlNode &gpu = xml->nodes[1];
  ASSERT_EQ(system.nAttrs, 2);
  EXPECT_STREQ(system.attrs[0].key, "version");
  EXPECT_STREQ(system.attrs[0].value, "1");
  EXPECT_STREQ(system.attrs[1].key, "name");
  EXPECT_STREQ(system.attrs[1].value, "FlagCX parser test");
  ASSERT_EQ(gpu.nAttrs, 3);
  EXPECT_STREQ(gpu.attrs[0].key, "dev");
  EXPECT_STREQ(gpu.attrs[0].value, "0");
  EXPECT_STREQ(gpu.attrs[1].key, "rank");
  EXPECT_STREQ(gpu.attrs[1].value, "1");
  EXPECT_STREQ(gpu.attrs[2].key, "note");
  EXPECT_STREQ(gpu.attrs[2].value, "contains \"double quotes\"");
}

TEST(XmlParser, AcceptsMaximumAttributeValueAndRejectsOverflow) {
  const std::string maxValue(MAX_STR_LEN, 'x');
  const std::string validContents =
      "<system version=\"1\"><gpu note=\"" + maxValue +
      "\"/></system>";

  flagcxXml *rawXml = nullptr;
  ASSERT_EQ(parseXmlText(validContents, &rawXml), flagcxSuccess);
  std::unique_ptr<flagcxXml, decltype(&std::free)> xml(rawXml, &std::free);
  ASSERT_EQ(xml->maxIndex, 2);
  EXPECT_EQ(std::string(xml->nodes[1].attrs[0].value).size(),
            static_cast<size_t>(MAX_STR_LEN));

  const std::string overlongValue(MAX_STR_LEN + 1, 'x');
  const std::string invalidContents =
      "<system version=\"1\"><gpu note=\"" + overlongValue +
      "\"/></system>";
  rawXml = nullptr;
  EXPECT_EQ(parseXmlText(invalidContents, &rawXml), flagcxInternalError);
  EXPECT_EQ(rawXml, nullptr);
}

TEST(XmlParser, RejectsUnterminatedQuotedAttribute) {
  const std::string contents =
      "<system version='1'><gpu note='unterminated></gpu></system>";
  flagcxXml *rawXml = nullptr;
  EXPECT_EQ(parseXmlText(contents, &rawXml), flagcxInternalError);
  EXPECT_EQ(rawXml, nullptr);
}

} // namespace
