import { Box, Heading, Image, Stack } from "@chakra-ui/react";
import React from "react";

export default function Logo() {
  return (
    <Box>
      <a href="https://lab.moffitt.org/Thieu/" target="_blank">
        <Stack direction={"row"} spacing={4} alignItems={"center"}>
          <Image
            src="/images/logo.jpg"
            alt="LAILab Logo"
            width={50}
            height={50}
          />
          <Heading fontSize="2xl" fontWeight={"bold"}>
            LAILab
          </Heading>
        </Stack>
      </a>
    </Box>
  );
}
