import "react-toastify/dist/ReactToastify.css";
import { Card, CardBody, Heading } from "@chakra-ui/react";
import { sendFeedback } from "../utils/sendFeedback";
import { apiBaseUrl } from "../utils/constants";
import { useColorModeValue } from "@chakra-ui/color-mode";
import * as chroma from "chroma.ts";

export type Source = {
  url: string;
  title: string;
};

export function SourceBubble({
  source,
  highlighted,
  lightMode,
  darkMode,
  onMouseEnter,
  onMouseLeave,
  runId,
}: {
  source: Source;
  highlighted: boolean;
  lightMode: string;
  darkMode: string;
  onMouseEnter: () => any;
  onMouseLeave: () => any;
  runId?: string;
}) {
  const bgColor = useColorModeValue(lightMode, darkMode);
  const LightHoverMode = chroma.color(bgColor).darker(0.5).css();
  const DarkHoverMode = chroma.color(bgColor).darker(0.5).css();
  const hoverBgColor = useColorModeValue(LightHoverMode, DarkHoverMode);

  return (
    <Card
      onClick={async () => {
        const sourceUrl = `${apiBaseUrl}/documents/${source.url}`;
        console.log("sourceUrl", sourceUrl, "runId", runId);
        window.open(sourceUrl, "_blank");

        // NOTE: runId is currently undefined; sendFeedback is not called
        if (runId) {
          await sendFeedback({
            key: "user_click",
            runId,
            value: source.url,
            isExplicit: false,
          });
        }
      }}
      backgroundColor={highlighted ? hoverBgColor : bgColor}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      cursor={"pointer"}
      alignSelf={"stretch"}
      height="100%"
      overflow={"hidden"}
    >
      <CardBody>
        <Heading size={"sm"} fontWeight={"normal"}>
          {source.title}
        </Heading>
      </CardBody>
    </Card>
  );
}
