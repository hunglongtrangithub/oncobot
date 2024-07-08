import { MouseEvent } from "react";
import { Heading, Card, CardHeader, Flex, Spacer } from "@chakra-ui/react";
import { useColorModeValue } from "@chakra-ui/color-mode";
import * as chroma from "chroma.ts";

export function EmptyState(props: {
  onChoice: (question: string) => any;
  lightMode: string;
  darkMode: string;
}) {
  const { onChoice, lightMode, darkMode } = props;
  const bgColor = useColorModeValue(lightMode, darkMode);
  const LightHoverMode = chroma.color(bgColor).darker(0.5).css();
  const DarkHoverMode = chroma.color(bgColor).darker(0.5).css();
  const hoverBgColor = useColorModeValue(LightHoverMode, DarkHoverMode);
  const handleClick = (e: MouseEvent) => {
    onChoice((e.target as HTMLDivElement).innerText);
  };

  const CustomCard = (props: { text: string }) => (
    <Card
      onMouseUp={handleClick}
      width={"48%"}
      backgroundColor={bgColor}
      _hover={{ backgroundColor: hoverBgColor }}
      cursor={"pointer"}
      justifyContent={"center"}
    >
      <CardHeader justifyContent={"center"}>
        <Heading
          fontSize="lg"
          fontWeight={"medium"}
          mb={1}
          textAlign={"center"}
        >
          {props.text}
        </Heading>
      </CardHeader>
    </Card>
  );

  return (
    <div className="rounded flex flex-col items-center max-w-full">
      <Heading
        fontSize="xl"
        fontWeight={"normal"}
        mb={1}
        marginTop={"10px"}
        textAlign={"center"}
      >
        Example Questions
      </Heading>
      <Flex marginTop={"25px"} grow={1} maxWidth={"800px"} width={"100%"}>
        <CustomCard text="What is the diagnosis of Fake Patient1?" />
        <Spacer />
        <CustomCard text="What is the treatment for Fake Patient2?" />
      </Flex>
      <Flex marginTop={"25px"} grow={1} maxWidth={"800px"} width={"100%"}>
        <CustomCard text="What is the diagnosis of Fake Patient3?" />
        <Spacer />
        <CustomCard text="What is the treatment for Fake Patient4?" />
      </Flex>
    </div>
  );
}
