export type RednotePost = {
  id: string;
  modelType: string;
  xsecToken: string | null;
  url: string;
  noteCard: {
    type: string | null;
    displayTitle: string | null;
    cover: {
      urlDefault: string | null;
      urlPre: string | null;
      url: string | null;
      fileId: string | null;
      width: number | null;
      height: number | null;
      infoList: Array<{
        imageScene: string | null;
        url: string | null;
      }>;
    };
    user: {
      userId: string | null;
      nickname: string | null;
      nickName: string | null;
      avatar: string | null;
      xsecToken: string | null;
    };
    interactInfo: {
      liked: boolean;
      likedCount: string | null;
      commentCount: string | null;
      collectedCount: string | null;
      sharedCount: string | null;
    };
    cornerTagInfo: Array<{
      type: string | null;
      text: string | null;
    }>;
    imageList: Array<{
      width: number | null;
      height: number | null;
      infoList: Array<{
        imageScene: string | null;
        url: string | null;
      }>;
    }>;
    video: {
      duration: number | null;
    };
  };
};
