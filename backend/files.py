export const _uploadToFileSearchStoreLogic = ai.defineFlow(
  {
    name: 'uploadToFileSearchStore',
    inputSchema: z.object({
      fileData: z.string(), // Base64 encoded file
      mimeType: z.string(),
      displayName: z.string().optional(),
    }),
    // ...
  },
  async ({fileData, mimeType, displayName}) => {
    const genAI = new GoogleGenAI({});

    // ... (logic to get/create store name)

    const blob = new Blob([Buffer.from(fileData, 'base64')], { type: mimeType });

    // This single call handles chunking and indexing!
    let operation = await genAI.fileSearchStores.uploadToFileSearchStore({
      file: blob,
      fileSearchStoreName: googleFileSearchStoreName,
      config: {
        displayName: displayName || 'uploaded-file',
        mimeType: mimeType,
      }
    });

    // We wait for the background indexing to complete
    while (!operation.done) {
      await new Promise((resolve) => setTimeout(resolve, 5000));
      const operationResult = await genAI.operations.get({ operation });
      operation = operationResult as any;
    }
       return { success: true };
  }
);

export const _chatWithFileSearchLogic = ai.defineFlow(
  // ... input: question, history
  async({question, history}) => {
    // We retrieve the store name where our files are indexed
    const storeSnap = await db.collection('obhqFileSearchStores').doc('obhqknowledge').get();
    const googleFileSearchStoreName = storeSnap.data()?.googleFileSearchStoreName;

    const response = await ai.generate({
      system: CHAT_WITH_FILE_SEARCH_SYSTEM_PROMPT,
      messages: [
        ...toGenkitMessages(history ?? []),
        {role: 'user', content: [{text: question}]},
      ],
      config: {
        tools: [
          {
            fileSearch: {
              fileSearchStoreNames: [googleFileSearchStoreName]
            }
          }
        ]
      }
    });

    return response.text;
  }
);