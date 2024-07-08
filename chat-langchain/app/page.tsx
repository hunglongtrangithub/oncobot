"use client";

import { ChakraProvider } from "@chakra-ui/react";
import App from "./App";

export default function Home() {
  return (
    <ChakraProvider>
      <App></App>
    </ChakraProvider>
  );
}
