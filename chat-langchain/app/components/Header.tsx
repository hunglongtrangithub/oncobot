import {
  Box,
  Button,
  Flex,
  Heading,
  Spacer,
  Stack,
  useColorModeValue,
} from "@chakra-ui/react";
import { MoonIcon, SunIcon } from "@chakra-ui/icons";

import { useDisclosure, useMediaQuery } from "@chakra-ui/react";
import {
  Drawer,
  DrawerBody,
  DrawerFooter,
  DrawerHeader,
  DrawerOverlay,
  DrawerContent,
  DrawerCloseButton,
} from "@chakra-ui/react";
import { IoMdMenu } from "react-icons/io";

import Footer from "./Footer";
import Logo from "./Logo";
import React from "react";

// import { Menu, MenuButton, MenuList, MenuDivider } from "@chakra-ui/react";
// import { Center } from "@chakra-ui/react";
// export function MobileMenu(props: {
//   title: string;
//   colorMode: string;
//   children?: React.ReactNode;
// }) {
//   const { title, colorMode, children } = props;
//   return (
//     <Menu>
//       <MenuButton
//         as={Button}
//         rounded={"full"}
//         variant={"link"}
//         cursor={"pointer"}
//         minW={0}
//       >
//         <IoMdMenu size="26px" />
//       </MenuButton>
//       <MenuList alignItems={"center"} justifyContent={"center"}>
//         <Box display="flex" flexDirection="column" alignItems="center" w="100%">
//           <Heading size="md">{title}</Heading>
//           <MenuDivider />
//           <Center>{children}</Center>
//           <MenuDivider />
//           <Footer colorMode={colorMode} />
//         </Box>
//       </MenuList>
//     </Menu>
//   );
// }

export function MobileDrawer({
  children,
  title,
  footer,
  width = "300px",
  placement = "right",
  p = 4,
}: {
  children?: React.ReactNode;
  title: string;
  footer: React.ReactNode;
  width?: string;
  placement?: "top" | "left" | "right" | "bottom";
  p?: number;
}) {
  const btnRef = React.useRef<HTMLButtonElement>(null);
  const { isOpen, onOpen, onClose } = useDisclosure();

  return (
    <Flex>
      <Button ref={btnRef} onClick={onOpen}>
        <IoMdMenu size="26px" />
      </Button>

      <Flex w={width} display={{ base: "none", md: "flex" }}>
        <Drawer
          isOpen={isOpen}
          placement={placement}
          onClose={onClose}
          finalFocusRef={btnRef}
        >
          <DrawerOverlay />
          <DrawerContent alignItems="center">
            <DrawerCloseButton alignSelf="end" mx={p} my={p} />
            <DrawerHeader my={p}>
              <Heading size="md">{title}</Heading>
            </DrawerHeader>
            <DrawerBody>{children}</DrawerBody>
            <DrawerFooter>{footer}</DrawerFooter>
          </DrawerContent>
        </Drawer>
      </Flex>
    </Flex>
  );
}
export const Header = (props: {
  titleText: string;
  lightMode: string;
  darkMode: string;
  toggleVoiceChat: () => void;
  isVoiceChatActive: boolean;
  toggleColorMode: () => void;
  colorMode: string;
}) => {
  const {
    titleText,
    lightMode,
    darkMode,
    toggleVoiceChat,
    isVoiceChatActive,
    toggleColorMode,
    colorMode,
  } = props;
  const [isLargerThanHeight] = useMediaQuery("(min-width: 100vh)");

  return (
    <Box
      bg={useColorModeValue(lightMode, darkMode)}
      as="nav"
      alignItems="center"
      justifyContent="space-between"
      w="100%"
      px={4}
      py={2}
      zIndex={1}
    >
      <Stack
        display={{ base: "none", md: "flex" }}
        direction={["column", "row"]}
        align="center"
        justify="space-between"
        wrap="wrap"
        w="100%"
        py={2}
      >
        <Logo />
        <Spacer />
        <Heading fontSize="2xl" fontWeight={"medium"} my={1}>
          {titleText}
        </Heading>
        <Spacer />
        <Stack
          direction={"row"}
          position="relative"
          spacing={2}
          maxWidth={"150px"}
        >
          <Button onClick={toggleVoiceChat}>
            {isVoiceChatActive ? "Voice Chat" : "Text Chat"}
          </Button>
          <Spacer />
          <Button onClick={toggleColorMode}>
            {colorMode === "light" ? <MoonIcon /> : <SunIcon />}
          </Button>
        </Stack>
      </Stack>
      <Stack
        display={{ base: "flex", md: "none" }}
        // direction={["column", "row"]}
        direction={"row"}
        align="center"
        justify="space-between"
        wrap="wrap"
        w="100%"
        py={2}
      >
        <Logo />
        <Spacer />
        <Stack direction={"row"} position="relative" spacing={2}>
          <Button onClick={toggleColorMode}>
            {colorMode === "light" ? <MoonIcon /> : <SunIcon />}
          </Button>
          <MobileDrawer
            title={titleText}
            footer={<Footer colorMode={colorMode} />}
            placement={isLargerThanHeight ? "right" : "top"}
          >
            <Button onClick={toggleVoiceChat}>
              {isVoiceChatActive ? "Voice Chat" : "Text Chat"}
            </Button>
          </MobileDrawer>
          {/* <MobileMenu title={titleText} colorMode={colorMode}>
            <Button key="voiceChatButton" onClick={toggleVoiceChat}>
              {isVoiceChatActive ? "Voice Chat" : "Text Chat"}
            </Button>
          </MobileMenu> */}
        </Stack>
      </Stack>
    </Box>
  );
};
